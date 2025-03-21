# Predição de Performance de Estudantes

Neste projeto, realizarei uma predição do desempenho de estudantes utilizando o dataset sintético Predict Student Performance do Kaggle. Esse conjunto de dados foi criado para representar fatores-chave do mundo real, como hábitos de estudo, padrões de sono, histórico socioeconômico e frequência às aulas.

Dataset: https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset

# Objetivo

O objetivo da predição é oferecer às escolas uma visão detalhada das rotinas e comportamentos que diferenciam os alunos com melhor e pior desempenho. Essa análise permitirá identificar estratégias eficazes para melhorar o rendimento acadêmico de todos os estudantes, promovendo um ambiente educacional mais equilibrado e produtivo.

Para alcançar esse objetivo, empregarei técnicas de aprendizado supervisionado, como Random Forest e Decision Tree , avaliando comparativamente o desempenho de cada modelo na tarefa de predição. Isso permitirá determinar qual abordagem é mais adequada para capturar as relações entre os fatores analisados e o desempenho dos alunos.

## Colunas

**Study Hours**

Descrição: Média de horas diárias dedicadas aos estudos.

**Sleep Hours**

Descrição: Média de horas diárias de sono.

**Socioeconomic Score**

Descrição: Uma pontuação normalizada (0-1) que indica o contexto socioeconômico do estudante.

**Attendance (%)**

Descrição: Percentual de aulas frequentadas pelo estudante.

**Grades (TARGET)**

Descrição: Pontuação final de desempenho do estudante, derivada da combinação de horas de estudo, horas de sono, pontuação socioeconômica e frequência.

## Relatório Análise Univariada

**Socioeconomic Score**: A variável tem uma distribuição concentrada em valores próximos a zero, como já dito antes que os dados já estão normalizados

**Study Hours**: Os estudantes parecem ter uma quantidade razoavelmente consistente de horas dedicadas aos estudos, sem grandes discrepâncias.

**Sleep Hours**: Os estudantes tendem a dormir por um número de horas bastante consistente, sem grandes diferenças individuais.

**Attendance (%):** A frequência de presença varia significativamente entre os estudantes. Alguns alunos frequentam aula quase sempre, enquanto outros têm presença irregular.

**Grades:** As notas dos estudantes variam consideravelmente. A maioria tem notas moderadas, mas há um grupo de alunos com desempenho superior.

Após análise inicial, observei que todas as colunas já estão normalizadas, o que elimina a necessidade de etapas adicionais de feature engineering relacionadas à padronização ou transformação dos dados.

No entanto, identifiquei a presença de outliers , principalmente nas variáveis Attendance (Frequência) e Grades (Notas) . Esses valores extremos são esperados e fazem sentido no contexto do dataset;

Em relação à Attendance , é natural que alguns alunos apresentem taxas de frequência significativamente mais altas do que a média, refletindo um engajamento excepcional com as atividades acadêmicas.
Já em Grades , a existência de notas muito acima da média pode ser explicada pela presença de estudantes com desempenho destacado, o que é consistente com a realidade educacional.
Esses outliers não parecem ser resultado de erros nos dados, mas sim características inerentes à distribuição das variáveis, e podem fornecer insights valiosos para análises futuras.

# Relatório da Análise Bivariada

![image](https://github.com/user-attachments/assets/10fb9984-0b9e-4ad4-ad12-d79945135359)

![image](https://github.com/user-attachments/assets/ed673442-2b09-475d-ac0a-0111570a0b5b)

A matriz de correlação e o gráfico de dispersão forneceram insights valiosos sobre as relações entre as variáveis no contexto do desempenho acadêmico. Destacou-se a importância do tempo dedicado ao estudo (Study Hours (0.81)) como o fator mais relevante para boas notas, seguido pela frequência de presença nas aulas (Attendance % ) e pelo índice socioeconômico (Socioeconomic Score ). Além disso, foi evidenciada a necessidade de equilíbrio entre estudo e descanso, bem como a influência de fatores socioeconômicos no desempenho. Essas informações podem orientar estratégias educacionais eficazes para melhorar o rendimento dos alunos.

# Separação de Dados em Treino e Teste e Pré Processamento

Foi utilizado um pipeline para organizar o fluxo de pré-processamento dos dados, garantindo a prevenção de data leakage (vazamento de informações). Dentro do pipeline, foram empregados os seguintes passos:

SimpleImputer : Aplicado para preencher possíveis valores nulos que possam surgir no conjuntos de dados, garantindo que o modelo não seja impactado por lacunas nos dados.

StandardScaler : Utilizado para normalizar as features, transformando-as para que tenham média igual a zero e desvio padrão igual a um. Essa etapa é essencial para algoritmos sensíveis à escala dos dados, como modelos lineares ou baseados em distância.

# Modelagem

A partir daqui decidimos criar dois modelos um de Random Forest e um de DecisionTreeRegressor para que eu possa avaliar qual modelo tem o melhor desempenho

![image](https://github.com/user-attachments/assets/586a47be-4f2c-4a1b-b4b1-6d3d5d0c69eb)

![image](https://github.com/user-attachments/assets/e34f9f12-8e27-40aa-b0f3-7d7ee80735b4)

Com base nos resultados obtidos, o modelo Random Forest apresentou um desempenho superior em relação ao Decision Tree na predição do desempenho dos estudantes. Isso pode ser observado pelos menores valores de erro absoluto médio (MAE = 0.8891), erro quadrático médio (MSE = 1.4414) e raiz do erro quadrático médio (RMSE = 1.2006), além de um coeficiente de determinação (R²) de 0.9809, indicando um excelente ajuste aos dados.

Já o modelo Decision Tree obteve métricas de erro significativamente mais altas (MAE = 1.3129, MSE = 3.5935, RMSE = 1.8957) e um R² inferior (0.9524), sugerindo um ajuste menos preciso.

Dessa forma, concluímos que o Random Forest é a melhor escolha para este problema, pois demonstra maior capacidade preditiva e generalização, reduzindo os erros e fornecendo previsões mais confiáveis sobre o desempenho dos estudantes.

# Validação cruzada com R²

Devido ao R² elevado, optei por realizar uma validação cruzada para garantir a confiabilidade do modelo. Um R² muito alto pode, em alguns casos, indicar que o modelo está se ajustando excessivamente aos dados de treino, capturando não apenas padrões reais, mas também ruídos e variações irrelevantes, o que pode levar ao overfitting.

![image](https://github.com/user-attachments/assets/3ece56a9-716e-44ac-97fd-146120ba8d39)

O modelo Random Forest apresentou um R² de 0.9809, indicando que ele explica 98.09% da variabilidade dos dados, o que sugere um excelente ajuste. Para confirmar sua generalização, foi aplicada validação cruzada, resultando em um R² médio de 0.9766, com um desvio padrão de 0.0029.

A pequena diferença entre os valores indica que o modelo não está sofrendo overfitting, pois o desempenho se mantém estável mesmo em diferentes subconjuntos dos dados. Além disso, as métricas de erro (MAE e RMSE) são baixas, reforçando a confiabilidade do modelo. Assim, ele pode ser considerado preciso e generalizável para novos dados.

# Gráfico de Dispersão para demonstrar visualmente o modelo RandomForest

![image](https://github.com/user-attachments/assets/7c82f3a7-2b96-40f2-ab46-f2775fb36b3e)

# Conclusão 

Neste projeto, empregamos técnicas de machine learning para prever o desempenho dos estudantes com base em um conjunto de dados sintético que simula variáveis reais, como hábitos de estudo e contexto socioeconômico. Ao comparar diferentes abordagens, constatamos que o modelo Random Forest apresentou resultados superiores à árvore de decisão, evidenciando maior precisão e consistência nas previsões.

O Random Forest não só gerou menores índices de erro, mas também demonstrou alta capacidade explicativa dos dados, o que se refletiu em valores de R² próximos de 1. A estabilidade dos resultados, verificada por meio da validação cruzada, reforça a confiança na capacidade do modelo de generalizar bem para novos cenários, minimizando o risco de overfitting.

Com base nos resultados obtidos, concluímos que o Random Forest se destacou como a estratégia mais eficaz para a predição de desempenho estudantil neste dataset. No entanto, devido ao número limitado de variáveis disponíveis, acredita-se que a inclusão de variáveis adicionais e significativas poderia enriquecer ainda mais a base de dados. A ausência dessas variáveis pode estar atenuando as diferenças de desempenho entre os modelos, tornando-os mais similares em termos de resultados.


