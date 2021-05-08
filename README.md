# Hackathon Itaú Asset 
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

Este repositório contém 3 arquivos .py e um notebook 
que ilustra a utilidade dos códigos modulares desenvolvidos. É neste 
notebook que encontramos a resolução das partes dos desafios propostos. Todos resolvidos 
com as funções dos códigos modulares.

## **Arquivos**

* ```optimization/markowitz.py```: arquivo que introduz a classe ```Markowitz``` e as 
  funções que servirão como base para as aplicações
* ```efficient_frontier.py```: arquivo que introduz a classe ```EfficientFrontier```, que utiliza
  as criações do arquivo ```markowitz.py``` para construir a Fronteira Eficiente 
  e as respostas esperadas das aplicações. 
* ```resolution.ipynb```: notebook o qual contém, de forma mais clara 
e consisa, a resolução de cada parte do desafio, assim como as aplicações desejadas.
* ```requirements.txt```: arquivo de texto que contém as bibliotecas e dependências para a utilização dos códigos. 

## Referências 
* *Mathematical Finantial Economics, a Basic Introduction.* I.Evstigneev, T.Hens, K.Schenk-Hoppé

