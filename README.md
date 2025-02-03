# Hackathon Itaú Asset 

## Feedback do Hackathon
1. ✅ Entregou pacote python conforme solicitado.
2. Não documentou o algoritmo apresentado ❌, mas documentou parcialmente o código.
3. A solução apresentada é incorreta (viola os vínculos) ❌.
4. A FE é calculada, mas o gráfico é específico do exemplo e não um método genérico.
5. ✅ Entregou corretamente estimação da matriz de covariância e fronteira eficiente do exemplo solicitado.
6. Entregou a recomendação de portfólio, mas há um erro ao usar a raiz quadrada do período da raiz quadrada da covariância (raiz quarta no fim) ❌.


## Usabilidade 
Clone este repositório e use um ambiente virtual conforme descrito abaixo.
```bash
$ cd hack_itau_quant
$ virtualenv venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt
```

Com o ambiente virtual python contendo as dependencia basta executar o python notebook com as respostas.

## Construção de um robô consultor 

Este repositório foi dedicado para as entregas e códigos, desenvolvidos pela equipe 
de finanças quantitativas do Turing.USP Quant, na competição de elaboração da Itaú Asset 
em parceria com a Bloomberg. 

O objetivo da competição constia na criação de um "robô consultor"
o qual aplica o algortimo de Markowitz para otimização de portfolio. 
Além disso, constrói o gráfico da fronteira eficiente de média-variância. 

Ademais, o projeto também inclui uma aplicação (Parte 2) do que foi desenvolvido na 
Parte 1. Nesta aplicação, consideramos o investimento em 4 cotas de fundos de 
investimento (dados tirados através da utilização da API do Bloomberg).

## Conteúdo do repositório 

Este repositório contém diversos arquivos .py e 3 notebooks
que ilustram a utilidade dos códigos modulares desenvolvidos. É neste 
notebook que encontramos a resolução das partes dos desafios propostos. Todos resolvidos 
com as funções dos códigos modulares.

## **Arquivos**

* ```optimization/markowitz.py```: arquivo que introduz a classe ```Markowitz``` e as 
  funções que servirão como base para as aplicações

* ```efficient_frontier.py```: arquivo que introduz a classe ```EfficientFrontier```, que utiliza
  as criações do arquivo ```markowitz.py``` para construir a Fronteira Eficiente 
  e as respostas esperadas das aplicações. 

* ```Resolution.ipynb```: notebook o qual contém, de forma mais clara 
e consisa, a resolução de cada parte do desafio, assim como as aplicações desejadas.

* ```Denosing.ipynb```: notebook que apresenta o módulo de denosing, comparando a utilização de um Markowitz com diferentes técnicas de denoising.

* ```Backtesting.ipynb```: notebook que realiza o backtesting de diferentes estratégias de otimização, rebalanceamndo uma carteira a cada 20 dias.



* ```requirements.txt```: arquivo de texto que contém as bibliotecas e dependências para a utilização dos códigos. 

## Referências 
* *Mathematical Finantial Economics, a Basic Introduction.* I.Evstigneev, T.Hens, K.Schenk-Hoppé

