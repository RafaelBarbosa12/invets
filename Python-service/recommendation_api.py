# Python-service/recommendation_api.py

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import logging
from typing import Dict, List, Tuple, Set

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class RecommendationService:
    def __init__(self, csv_path: str = './dataset/carteiras_btg.csv', min_users_for_item: int = 5, K_neighbors: int = 20):
        """Inicializa o serviço de recomendação carregando e processando os dados."""
        self.csv_path = csv_path
        self.min_users_for_item = min_users_for_item
        self.K_neighbors = K_neighbors
        self.df = None
        self.portfolio_pct = None
        self.item_neighbors = {}
        # self.usuarios_alvo: List[str] = [] # Esta lista não será usada para a lógica de "conformidade" via DB lookup
        self.user_item = None
        self.map_tipo_para_classe: Dict[str, str] = {} # Storing the mapping

        try:
            self.load_and_process_data()
            logger.info("Serviço de recomendação inicializado com sucesso.")
        except FileNotFoundError:
            logger.error(f"Erro: O arquivo CSV '{self.csv_path}' não foi encontrado. Certifique-se de que o caminho está correto.")
            raise
        except Exception as e:
            logger.error(f"Erro ao inicializar serviço: {str(e)}")
            raise

    def load_and_process_data(self):
        """Carrega e processa os dados do CSV, espelhando a lógica do seu modelo."""
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Dados carregados: {self.df.shape}")
        
        # Mapear tipos de ativos para classes
        unique_tipos = self.df['Tipo_de_Ativo'].unique()
        
        rf_keywords = ['RF', 'CDB', 'LCI', 'LCA', 'CRI', 'CRA', 'DEBENTURE', 'TESOURO', 'BOND', 'DÍVIDA']
        rv_keywords = ['ACAO', 'AÇÃO', 'ETF', 'FII', 'BDR', 'REIT', 'FIDC', 'FIAGRO', 'STOCK', 'EQUITY', 'RV']
        
        for t in unique_tipos:
            upper = t.upper()
            if any(k in upper for k in rf_keywords):
                self.map_tipo_para_classe[t] = 'RF'
            elif any(k in upper for k in rv_keywords):
                self.map_tipo_para_classe[t] = 'RV'
            else:
                self.map_tipo_para_classe[t] = 'OUTROS'
        
        self.df['Classe_Ativo'] = self.df['Tipo_de_Ativo'].map(self.map_tipo_para_classe)
        
        # Calcular percentuais por conta & classe (ainda útil para o self.df para o Item-KNN)
        self.portfolio_pct = (
            self.df.pivot_table(index='Conta', columns='Classe_Ativo', values='Financeiro', aggfunc='sum')
                .fillna(0)
        )
        self.portfolio_pct = self.portfolio_pct.div(self.portfolio_pct.sum(axis=1), axis=0) * 100
        
        # Junta perfil (ainda útil para o self.df para o Item-KNN)
        perfil = self.df.groupby('Conta')['Perfil_da_carteira'].first()
        self.portfolio_pct = self.portfolio_pct.join(perfil)
        
        # Verificar conformidade (ainda útil para o self.df para o Item-KNN)
        self.portfolio_pct['em_conformidade'] = self.portfolio_pct.apply(self._verifica_conformidade, axis=1)
        
        # Usuários fora de conformidade (mantido para fins de preparar o modelo Item-KNN,
        # mas não para lookup dinâmico do usuário da requisição)
        # self.usuarios_alvo = self.portfolio_pct[~self.portfolio_pct['em_conformidade']].index.tolist()
        # logger.info(f"Total de usuários fora de conformidade no dataset: {len(self.usuarios_alvo)}")

        # Preparar recomendações Item-KNN
        self._prepare_item_knn_model()

    def _verifica_conformidade(self, row: pd.Series) -> bool:
        """Verifica se a carteira está em conformidade com o perfil."""
        perfil = row['Perfil_da_carteira']
        rf = row.get('RF', 0)
        rv = row.get('RV', 0)
        outros = row.get('OUTROS', 0)
        
        if perfil == 'Conservador':
            return (rf >= 90) and (rv + outros <= 10)
        elif perfil == 'Moderado':
            return (rf >= 60) and (rv + outros <= 40)
        elif perfil in ['Sofisticado', 'Arrojado', 'Sofisticado/Arrojado']:
            return (rv >= 70) and (rf + outros <= 30)
        else:
            return False

    def _prepare_item_knn_model(self):
        """Prepara a matriz de similaridade para recomendações (Item-KNN)."""
        
        # Filtra usuários em conformidade para construir a matriz usuário-item
        # Usamos df_ok porque o Item-KNN aprende com dados de carteiras "boas"
        df_ok = self.df[self.df['Conta'].isin(self.portfolio_pct[self.portfolio_pct['em_conformidade']].index)]

        if df_ok.empty:
            logger.warning("No compliant users found to build Item-KNN model. Item recommendations may be limited.")
            self.user_item = pd.DataFrame()
            self.item_neighbors = {}
            return

        # Financeiro por conta e ativo
        user_item_raw = (
            df_ok.pivot_table(index='Conta', columns='Nome_Ativo', values='Financeiro', aggfunc='sum')
                .fillna(0)
        )

        sum_rows = user_item_raw.sum(axis=1)
        # Avoid division by zero for users with zero total value
        user_item_raw = user_item_raw.div(sum_rows.replace(0,1), axis=0) 
        
        # Remove ativos raros
        item_counts = (user_item_raw > 0).sum(axis=0)
        self.user_item = user_item_raw.loc[:, item_counts >= self.min_users_for_item]
        
        if self.user_item.empty:
            logger.warning("No items left after filtering rare items. Item recommendations will be empty.")
            self.item_neighbors = {}
            return

        logger.info(f"Matriz Usuário-Item para Item-KNN preparada. Shape: {self.user_item.shape}")

        # Matriz esparsa item×usuário
        item_user_mat = sp.csr_matrix(self.user_item.T.values)
        
        # Similaridade item×item (cosseno)
        sim = cosine_similarity(item_user_mat)
        
        # Top-K vizinhos por item
        topk = np.argsort(-sim, axis=1)[:, 1:self.K_neighbors+1] 
        
        # Dicionário: item -> vizinhos
        item_idx_to_name = dict(enumerate(self.user_item.columns))
        self.item_neighbors = {
            item_idx_to_name[i]: [item_idx_to_name[j] for j in topk[i]]
            for i in range(sim.shape[0])
        }
        logger.info(f"Item-KNN model trained with K={self.K_neighbors}. First few neighbors for 'ACOES BRASIL': {self.item_neighbors.get('ACOES BRASIL', 'N/A')}")


    def get_profile_recommendation_and_breakdown(self, assets: List[Dict]) -> Dict:
        """
        Calcula o breakdown do portfólio e o perfil de risco com base
        na lista de ativos fornecida.
        """
        try:
            # logger.info(f"Received assets for profile calculation: {assets}") # Redundante com log da rota
            if not assets:
                raise ValueError("Nenhum ativo fornecido para cálculo de perfil.")

            class_totals = {'RF': 0, 'RV': 0, 'OUTROS': 0}
            total_value = 0

            for asset in assets:
                asset_type = asset.get('assetType', '')
                financial_value = float(asset.get('financialValue', 0))
                
                upper_asset_type = asset_type.upper()
                asset_class_found = 'OUTROS' 
                
                rf_keywords = ['RF', 'CDB', 'LCI', 'LCA', 'CRI', 'CRA', 'DEBENTURE', 'TESOURO', 'BOND', 'DÍVIDA']
                rv_keywords = ['ACAO', 'AÇÃO', 'ETF', 'FII', 'BDR', 'REIT', 'FIDC', 'FIAGRO', 'STOCK', 'EQUITY', 'RV']

                if any(k in upper_asset_type for k in rf_keywords):
                    asset_class_found = 'RF'
                elif any(k in upper_asset_type for k in rv_keywords):
                    asset_class_found = 'RV'
                
                class_totals[asset_class_found] += financial_value
                total_value += financial_value
                # logger.debug(f"Processing asset: {asset_type}, class: {asset_class_found}, value: {financial_value}") 

            # Calculate percentages
            portfolio_breakdown = {}
            if total_value > 0:
                for asset_class, value in class_totals.items():
                    portfolio_breakdown[asset_class] = round((value / total_value) * 100, 2)
            else:
                portfolio_breakdown = {'RF': 0, 'RV': 0, 'OUTROS': 0}

            # Determine calculated profile based on the percentages
            calculated_profile = "Não Classificado"
            
            rf_pct = portfolio_breakdown.get('RF', 0)
            rv_pct = portfolio_breakdown.get('RV', 0)
            outros_pct = portfolio_breakdown.get('OUTROS', 0)

            if rf_pct >= 90:
                calculated_profile = "Conservador"
            elif rf_pct >= 60 and (rv_pct + outros_pct) <= 40:
                calculated_profile = "Moderado"
            elif rv_pct >= 70 and (rf_pct + outros_pct) <= 30:
                calculated_profile = "Arrojado"
            else:
                calculated_profile = "Misto/Não Classificado" # Adicionado para perfis que não se encaixam claramente

            return {
                'calculated_profile': calculated_profile,
                'portfolio_breakdown': portfolio_breakdown
            }

        except ValueError as e:
            logger.warning(f"Erro de validação ao processar cálculo de perfil: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Erro inesperado ao processar cálculo de perfil: {str(e)}")
            raise

    def recomendar_ativos(self, user_id: str, assets_currently_held: Set[str], N: int = 5) -> List[Tuple[str, int]]:
        """
        Recomenda ativos para um 'user_id' (apenas para logging) e um conjunto de ativos
        já possuídos (passados como argumento), usando o modelo Item-KNN.
        Não consulta banco de dados externo ou o dataframe 'self.df' para ativos possuídos.
        """
        logger.info(f"Iniciando recomendação de ativos para {user_id}. Ativos possuídos (input): {assets_currently_held}")
        
        scores = {}
        
        # Agrega votos de vizinhos para cada ativo possuído
        if not self.item_neighbors:
            logger.warning("Item-KNN model not initialized or empty. Cannot recommend assets.")
            return []

        for ativo in assets_currently_held:
            # Garante que o ativo existe no modelo Item-KNN
            if ativo in self.item_neighbors:
                vizinhos = self.item_neighbors.get(ativo, [])
                for v in vizinhos:
                    if v in assets_currently_held:  # Ignora ativos que o usuário já possui
                        continue
                    scores[v] = scores.get(v, 0) + 1  # Soma os votos
            else:
                logger.debug(f"Ativo '{ativo}' não encontrado nos vizinhos do Item-KNN (não há vizinhos similares no histórico).")
        
        # Retorna os N melhores ativos por pontuação
        recommendations = sorted(scores.items(), key=lambda x: -x[1])[:N]
        logger.info(f"Recomendações geradas para {user_id}: {recommendations}")
        return recommendations


# --- Instância Global e Rotas Flask ---

# Instância global do serviço
# Certifique-se de que 'carteiras_btg.csv' está no diretório correto './dataset/'
recommendation_service = RecommendationService(csv_path='./dataset/carteiras_btg.csv')


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde."""
    return jsonify({
        'status': 'healthy',
        'service': 'recommendation-api',
        'model_status': 'loaded' if recommendation_service.df is not None else 'failed to load'
    })

@app.route('/api/recommendation/profile', methods=['POST'])
def get_profile_recommendation_route():
    """
    Endpoint para recomendar um perfil com base em um portfólio fornecido,
    e ativos recomendados se o perfil atual divergir do calculado.
    Espera um JSON com 'userId', 'currentProfile' e 'assets' (lista de dicionários).
    
    Ex: {
      "userId": "cliente123",
      "currentProfile": "Moderado",
      "assets": [
        {"assetType": "CDB DI", "financialValue": 80000},
        {"assetType": "ACOES BRASIL", "financialValue": 5000}
      ]
    }
    """
    data = request.get_json()
    
    # Validação dos campos essenciais
    if not data:
        logger.warning("Dados de entrada vazios na requisição /api/recommendation/profile.")
        return jsonify({'error': 'Corpo da requisição vazio.'}), 400
    
    user_id = data.get('userId', 'default_user') # ID do usuário, para logging
    current_profile_input = data.get('currentProfile') # Perfil atual informado pelo usuário
    assets_data = data.get('assets', []) # Ativos do usuário

    if not current_profile_input:
        logger.warning(f"currentProfile não fornecido para {user_id}.")
        return jsonify({'error': 'O campo "currentProfile" é obrigatório.'}), 400
    
    if not assets_data or not isinstance(assets_data, list):
        logger.warning(f"Assets inválidos ou vazios na requisição para {user_id}.")
        return jsonify({'error': 'O campo "assets" é obrigatório e deve ser uma lista de ativos.'}), 400

    logger.info(f"Request for {user_id}. Current Profile: {current_profile_input}, Assets: {assets_data}")

    try:
        # 1. Calcular o perfil e o breakdown da carteira com base nos ativos fornecidos
        profile_calc_result = recommendation_service.get_profile_recommendation_and_breakdown(assets_data)

        calculated_profile = profile_calc_result['calculated_profile']
        portfolio_breakdown = profile_calc_result['portfolio_breakdown']
        
        asset_recommendations_for_profile = []
        
        # 2. Comparar o perfil informado com o perfil calculado para decidir se recomenda ativos
        if current_profile_input != calculated_profile:
            logger.info(f"Perfil atual ({current_profile_input}) difere do perfil calculado ({calculated_profile}) para {user_id}. Gerando recomendações de ativos.")
            
            # Extrair os nomes dos ativos que o usuário já possui da requisição
            assets_owned_names = {asset.get('assetType') for asset in assets_data if asset.get('assetType')}
            
            # Chamar a função de recomendação de ativos, passando os ativos possuídos
            raw_asset_recs = recommendation_service.recomendar_ativos(user_id, assets_owned_names, N=5) 
            
            asset_recommendations_for_profile = [
                {'assetName': asset, 'score': score} 
                for asset, score in raw_asset_recs
            ]
        else:
            logger.info(f"Perfil atual ({current_profile_input}) é igual ao perfil calculado ({calculated_profile}) para {user_id}. Nenhuma recomendação de ativo adicional.")


        response_data = {
            'calculatedProfile': calculated_profile, # Nome do campo ajustado para clareza
            'portfolioBreakdown': portfolio_breakdown
        }

        # Adicione as recomendações de ativos APENAS se elas existirem
        if asset_recommendations_for_profile:
            response_data['assetRecommendations'] = asset_recommendations_for_profile

        logger.info(f"Successfully processed profile recommendation. Final response for {user_id}: {response_data}")
        return jsonify(response_data), 200

    except ValueError as e:
        logger.warning(f"Erro de validação na requisição /api/recommendation/profile para {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Erro interno inesperado ao processar a recomendação de perfil para {user_id}:")
        return jsonify({'error': f'Erro interno ao processar a recomendação de perfil: {str(e)}'}), 500

# --- Tratamento de Erros Globais ---
@app.errorhandler(404)
def not_found(e):
    logger.warning(f"Endpoint não encontrado: {request.path}")
    return jsonify({'error': 'Endpoint não encontrado'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Erro interno do servidor capturado pelo errorhandler 500:")
    return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    # No modo de produção, debug=False é essencial por segurança.
    app.run(host='0.0.0.0', port=5000, debug=True)