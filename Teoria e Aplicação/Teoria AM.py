# Aqui vamos dar início ao uso de Aprendizado de Máquina, indo dos conceitos básico até às aplicações mais específicas
# O objetivo é capacitar teórica e praticamente para o uso de ML em situações da área de saúde

# Primeiro de tudo, o quê é aprendizado de máquina?
# Aprendizado de máquina é um ramo da inteligência artificial que permite que os computadores aprendam.
# Eles analisam centenas, milhares, milhões de dados a fim de aprender e melhorarem a si mesmos sem necessitar de programação
# Por exemplo, um computador pode aprender a identificar uma doença sem ter sido programado com os parâmetros da doença

# Existem três tipos principais de formas que um algoritmo de ML pode aprender:
'''
    1 - Aprendizado Supervisionado:
        Quando o modelo é treinado usando um conjunto de dados que já tem resposta correta. É como se estivéssemos ensinando
        com um gabarito, e ao final podemos ver quantas respostas ele de fato acertou.
        Assim, depois de treinar um modelo usando esse conjuntos de dados em que existe respostas corretas e incorretas,
        podemos analisar como ele se sairia na vida real.
           # Exemplos de uso: Detecção de doenças em exames.
    
    2 - Aprendizado Não-Supervisionado:
        Quando o modelo é treinado usando um conjunto de dados que não tem necessariamente uma resposta, e ele precisa
        encontrar estruturas e padrões por conta própria.
        Assim, pode ser usado para agrupar dados com base em padrões ou simplificar esses dados.
           # Exemplos de uso: Clustering(Agrupamento) de clientes por comportamento de compras. 
                                    "Quem comprou isso também comprou..."
                                    
    3 - Aprendizado por Reforço:
        Quando um "agente" interage com o ambiente e aprende a tomar decisões executando ações e recebendo feedback (pontos).
        Nesse caso, ele aprende a melhor sequência de ações para maximizar a recompensa total a longo prazo.
            # Exemplos de uso: Robótica, e IA de jogos.
'''



# Existem diversos métodos de Aprendizado de Máquina
# Estaremos trabalhando principalmente com modelos preditivos
# Dentre os métodos de modelos preditivos, temos os Métodos Baseados em Distâncias (que não vamos estudar por agora)
# Os Métodos Probabilísticos, os Métodos Simbólicos, Conexionistas e Maximização de Margens
# Estaremos focando nos Métodos Probabilísticos, Simbólicos e Conexionistas, já que são mais utilizados na Medicina

'''
######## MÉTODOS PROBABILÍSTICOS ########

#### APRENDIZADO BAYESIANO

    Os métodos probabilísticos Bayesianos assumem que a probabilidade de um evento A acontecer dado um evento B não depende
    somente da relação de A e B, mas também da probabilidade de observar A independentemente de observar B.
    
    Assim, surge a fórmula:
                                    P(A|B) = (P(B|A) x P(A))/P(B)
    Sendo P(A|B) a probabilidade de A ocorrer dado B, P(A) a probabilidade de A ocorrer, P(B) a probabilidade de B ocorrer e
    P(B|A) a probabilidade de B ocorrer sendo A verdadeiro.
    
    O que isso significa?
    Vamos dar um exemplo:
    
    Suponhamos o seguinte cenário:
    
        DOENÇA X: Presente em 8% dos casos, ausente em 92% dos casos
        
        Teste da Doença X:       Positivo            Negativo
            Doença Presente        75%                  25%
            Doença Ausente          4%                  96%
            
        Ou seja, a doença tem uma taxa de incidência de 8%.
        A probabilidade de um verdadeiro positivo (o teste dar positivo quando a doença está presente) é 75% (SENSIBILIDADE)
        A probabilidade de um verdadeiro negativo (o teste dar negativo quando a doença está ausente) é 96% (ESPECIFICIDADE)
        
        Podemos, então, fazer as seguintes perguntas (e calculá-las):
        1 - Qual a probabilidade de um teste dar positivo?
            P(Teste = Positivo) = (75% * 8%) + (4% * 92%) = 9,68%
        
        2 - Qual a probabilidade de um teste dar negativo?
            P(Teste = Negativo) = (25% * 8%) + (96% * 92%) = 90,32%
        
        3 - Se der positivo, qual a probabilidade da doença estar presente?
            P(Doença = Presente | Teste = Positivo) = (P(Teste = Positivo | Doença = Presente) * P(Doença = Presente))/P(Teste = Positivo)
                = (75% * 8%)/(9,68%) = ~61,98%
                
        4 - Se der negativo, qual a probabilidade da doença estar ausente?
            P(Doença = Ausente | Teste = Negativo) = (P(Teste = Negativo | Doença = Ausente) * P(Doença = Ausente))/P(Teste = Negativo)
                = (96% * 92%)/(90,32%) = ~97,79%
                
        5 - Se der positivo, qual a probabilidade da doença estar ausente?
            P(Doença = Ausente | Teste = Positivo) = (P(Teste = Positivo | Doença = Ausente) * P(Doença = Ausente))/P(Teste = Positivo)
                = (4% * 92%)/(9,68%) = ~38,02% (note que isso é o complemento de 3)
                
        6 - Se der negativo, qual a probabilidade da doença estar presente?
            P(Doença = Presente | Teste = Negativo) = (P(Teste = Negativo | Doença = Presente) * P(Doença = Presente))/P(Teste = Negativo)
                = (25% * 8%)/(90,32%) = ~2,21% (note que isso é o complemento de 4)
        
        
    Caso estejamos procurando a opção com maior probabilidade, podemos descartar o denominador das equações. Isso, porque estamos
    dividindo todas as equações pelo mesmo valor.
    

#### MÉTODO NAIVE BAYES

    O método "ingênuo Bayes" utiliza o teorema de Bayes assumindo uma coisa: que todas as características independem de si.
    Por isso, é um método "ingênuo"; mas ele funciona muito bem, mesmo para classificações complexas.
    
    Como ele funciona:
        P(Classe | Características) ∝ P(Classe) * P(Dado1|Classe) * P(Dado2|Classe) * ...
    Onde ∝ = é proporcional à
    
    Exemplo: 
        Pense que estamos classificando um email em spam ou não spam.
        Na fase de treinamento, o algoritmo lê milhares de emails. Ele nota que a palavra grátis aparece em 90% dos emails de
        spam, e 1% dos emails normais. Ele nota que a palavra dinheiro aparece 85% em emails de spam, e 2% nos emails normais.
        Ele faz isso para cada palavra ("reunião", "dinheiro", "cripto"...)
        Quando chega a fase de testes, ao receber um novo email, ele faz a probabilidade dele ser spam vs. a probabilidade de 
        não ser spam.
        E para calcular isso, ele utiliza a fórmula ali em cima. Nesse caso, supondo que queremos saber só com base na palavra
        "grátis":
        P(Spam | Grátis) ∝ P(Spam) * P(Grátis | Spam)
        
        Para outras palavras:
        P(Spam | Grátis, Dinheiro, Cripto) ∝ P(Spam) * P(Grátis | Spam) * P(Dinheiro | Spam) * P(Cripto | Spam)
        
        E assim por diante.
        
    Pelo fato de serem multiplicações simples, o método Naive Bayes é feito quase instantaneamente, mesmo para grande quantidades
    de dados.
    
    Note, no entanto, que o método Naive Bayes não serve para medir a codependência entre dois ou mais atributos. Novamente,
    ele analisa como se os atributos fossem independentes entre si. 
    Dois atributos são independentes quando P(X|Y) = P(X), o que implica em P(X,Y) = P(X) * P(Y)
    
    Assim, nós temos que resolver esse problema: e assim nascem as Redes Bayesianas
    
#### REDES BAYESIANAS PARA CLASSIFICAÇÃO

    As Redes Bayesianas utilizam o conceito de independência condicional para obter um equilíbrio entre um número de parâmetros
    para calcular e a representação de dependência entre as variáveis. 
    
    Nesses grafos:
    As Redes Bayesianas são uma tupla (S, Θ). S são a rede de Grafos Acíclicos Dirigidos, enquanto Θ são tabelas probabilísticas.
    1-Grafos Acíclicos Dirigos: São grafos formados por nós e setas (com direção e sentido) e você nunca vai seguir as setas e 
    acabar voltando pro ponto de partida (ou seja, não existem loops).
            - Nós(Círculos): Representam variáveis ou eventos
            - Setas(Arcos): Representam influência direta. Se há uma seta de A para B, então A é pai de B e A influencia a
                probabilidade de B acontecer.
            Os grafos dizem quem influencia quem, mas não diz quanto.
    
    2-Tabela de Probabilidade Condicional: Para cada nó no gráfico, existe uma tabela dentro dele. Ela representa o "peso" ou
        a força da influência. Ela contém os números exatos.
        
    Classificadores que usam a rede Bayesiana famosos é o TAN — em que cada atributo pode ser pai de somente um atributo — e o BAN,
    que há no máximo duas dependências entre atributos.
    
    Assim sendo, o cálculo de probabilidade fica algo do tipo:
    NAIVE BAYES = P(Classe) * P(X_1 | Classe) * P(X_2 | Classe) * P(X_3 | Classe) * ...
    TAN = P(Classe) * P(X_1 | Classe) * P(X_2|X_1, Classe) * P(X_3|X_2, Classe) * P(X_4|X_3, Classe) * ...
    BAN = P(Classe) * P(X_1 | Classe) * P(X_2|X_1, Classe) * P(X_3|X_2,X_1, Classe) * P(X_4|X_3, Classe) * ...
            
#### REGRESSÃO LOGÍSTICA

    Regressão logística é uma técnica estatística usada para modelar a probabilidade de uma variável dependente binária (isso é,
    uma variável com dois possíveis resultados, como 0 e 1, verdadeiro e falso, sucesso e falha)
    A Regressão Logística, ao contrário da linear (que é usada para prever valores contínuos), é usada para prever uma
    probabilidade que, por sua vez, pode ser transformada em linear.
    
    A função logística utiliza a seguinte fórmula:
    
                    P = o(z) = 1/(1 + e^(-z))
                    
        Onde P = probabilidade de ser 1 (ou seja, de ser um acerto)
        
        Assim sendo, z = log(p/(1 - p)) = b0 + b1*x1 + b2*x2 + x3*x3 + ... + bn*xn
        
    Esse p/(1 - p) chamamos de "odds". Odds nos diz sobre a proporção de 1 acontecer em relação a 0.
    A exemplo, seja p = 0.8 (80%). Temos que p/(1 - p) = 0.8/0.2 = 4, ou seja, a chance do evento ocorrer é quatro vezes maior
    que a chance de não ocorrer. E de fato, como p = 80%, !p = 20% e 80% = 20% * 4. Chamamos isso de "Odds Ratio". Isso é muito
    utilizado em epidemiologia.
    
    Por sua vez, a combinação linear b0 + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn é formada por "predictors".
        -x1, x2, x3... são as variáveis. São as características do paciente, em dados numéricos.
        -Chamamos b0 de "intercept". Ele nada mais é que a probabilidade base. Assim, se todas as variáveis forem 0, então temos
            que z = b0 e, portanto, p = o(b0).
        -b1, b2, b3... são os coeficientes das variáveis. Eles são fixos, e servem como pesos de cada variável. Se você aumentar
            x1 em 1 unidade, então z vai aumentar em b1. Mas como z é o "log odds", isso significa que odds são multiplicados por
            e^b1. 
                Ex: Imagine que b_idade = 0.7 => e^b_idade = e^(0.7) = 2.01
                    Isso significa que, para cada uma unidade que aumentemos em x_idade, a "odds" de ter a doença duplica.
'''

# Agora, passamos para os Métodos Simbólicos

'''
#### ÁRVORES DE DECISÃO E REGRESSÃO
    Uma árvore de decisão usa a estratégia de "dividir e conquistar" de modo a resolver um problema de decisão
    Assim, um problema complexo é dividido em problemas mais simples, e esses problemas mais simples são divididos recursivamente
    em problemas mais simples.
    
    Formalmente, uma árvore de decisão é um grafo direcionado acíclico em que cada nó ou é um nó de divisão, com dois ou mais
    sucessores, ou é um nó folha. Existem, também, o nó raiz, que é um nó de divisão cuja escolha de "pergunta" é mais importante:
    
        1 - Um nó folha é rotulado com uma função. No nó-folha, são consideradas apenas os valores da variável-alvo nos exemplos
        que chegam a um nó folha. Em problemas de classificação, e assumindo a função de custo 0-1, essa constante é a moda. Em
        problemas de regressão, a constante que minimiza a função de custo do erro médio quadrático é a média, enquanto para a
        função de custo do desvio absoluto é a mediana. 
    
        2 - Um nó de divisão contém um teste condicional baseado nos valores do atributo. Na proposta padrão, os testes são
        univariados; as condições envolvem um único atributo e valores no domínio desse atributo.
    
        3 - Um nó raiz contém o primeiro teste condicional, que divide todo o dataset. Normalmente, faz-se logo de cara a
        pergunta mais divisiva.
        
    Um exemplo de árvore de decisão simples utilizada para prever diabetes:
                                glicose > 140?
                                /        \
                              sim         não
                            IMC > 30?     saudável
                            /      \          
                        sim       não           
                     diabetes   saudável
    
    O modelo vai fazendo perguntas até chegar em uma decisão final, e cada pergunta usa uma variável.
    
    A criação de uma árvore de decisão é feita por um processo indutivo/recursivo. Ou seja, a criação de uma árvore de decisão
    é feita em uma função que chama a si mesma, até que chegue em um certo critério de parada. Quando chega nesse critério, é
    retornado um nó folha rotulado com a constante que minimiza a função custo.
    
    # Como a árvore decide qual pergunta fazer?
    
    A regra de divisão é guiada por uma medida de "goodness of split", que indica quão bem um certo atributo discrimina as
    classes; assim, seleciona-se um atributo que maximiza essa medida. 
    
    Assim, nasce-se um atributo chamado "impureza". É uma medida de aleatoriedade ou mistura de classes dentro de um conjunto
    de dados. A medida vai de 0 a 1, e a regra de divisão é feita de forma que os nós-filhos tenham a maior quantidade de pureza
    possível (ou seja, a menor impureza, a impureza mais próxima de 0 possível).
    
    Para calcular a impureza em algoritmos, existem duas formas principais de calcular a impureza, sendo elas o ÍNDICE DE GINI
    e a ENTROPIA. 
        1 - IMPUREZA DE GINI
            G = 1 - (p1² + p2² + p3² + p4² + ... pn²)
            
            Onde n = número total de classes, e pi(i indo de 1 a n) = proporção da classe i no nó atual.
            
        2 - ENTROPIA
            Entropia(S) = -(p1*Log(p1) + p2*Log(p2) + p3*Log(p3) + ... + pn*Log(pn))
            
            Onde S é o conjunto atual de nós, n é o número de classes únicas no conjunto, pi a proporção da classe i no nó atual
            e Log é logaritmo na base 2.

    # ÁRVORES DE REGRESSÃO VS ÁRVORES DE CLASSIFICAÇÃO
    
    Existem esses dois tipos principais de árvores.
    
        1 - Árvores de Classificação: Usadas quando o resultado que queremos é uma categoria (como sim/não e coisa do tipo).
            Ex: Árvores que respondam perguntas como "Tem câncer ou não", "Tem diabetes ou não", "Pneumonia ou normal" e etc.
            Normalmente, suas saídas são uma classe ("Doença", "Normal", "Sim", "Não"), mas também podem ser um valor numérico
            (por exemplo, "Probabilidade de Pneumonia: 0.73")
            
        2 - Árvores de Regressão: Usadas quando o resultado que queremos é um número contínuo. A exemplo, quando queremos estimar
            o tempo de sobrevivência, o nível da glicose ou o tempo de internação.
            Ex: "Idade > 70?" => Sim: "Tempo Médio de Internação: 12 dias", Não: "Tempo médio: 4 dias".
            
#### ESTRATÉGIAS DE PODA

    A poda é o processo mais importante do processo de construção da árvore (pelo menos em domínios com ruídos). Isso, pois
    árvores tem uma grande tendência a "overfitting", isso é, memorizar dados ao invés de aprender. Isso acaba fazendo uma árvore
    gigante e aprendendo ruídos. Podar árvores é especialmente importante pois os nós mais profundos acabam tendo pouca
    importância, visto que pouquíssimos casos chegam lá. Nós mais profundos refletem o overfitting e aumentam o erro em variância
    com o classificador. Além disso, árvores grandes tornam-nas difíceis de entender.
    
    Existem dois tipos de poda: Pré-Poda e Pós-Poda.
    
        1 - PRÉ-PODA: A Pré-Poda implica criar regras de parada que previnem a construção daqueles ramos que não parecem melhorar
            a precisão preditiva da árvore. Também salva tempo, já que não há tempo perdido construindo uma estrutura não usada
            na árvore final. Dentre estratégias de paradas, as duas universalmente aceitas são 1 - Quando todas as observações
            alcançando um nó pertencem à mesma classe e 2 - Todas as observações chegando a um nó tem o mesmo vetor de
            características (mas não necessariamente a mesma classe). Poderíamos, também, adicionar um limite de profundidade, ou
            um mínimo de pacientes por nó.
                
        2 - PÓS-PODA: O método mais comum de poda de árvores que, apesar de mais lento, é mais confiável (Quinlan, 1988). Assim,
            escolhemos uma subárvore, normalmente baseado na medida de erro estático e o erro de backed-up. O erro de backed-up
            é a soma das classificações incorretas de todas as subárvores do nó-corrente, enquanto o estático é o erro do sistema
            depois dele se estabilizar. Quando o erro de backed-up se torna maior ou igual ao erro estático, então o nó é trocado
            por um nó-folha com a classe majoritária do nó.
'''

# MÉTODOS CONEXIONISTAS

'''
#### REDES NEURAIS ARTIFICIAIS
    Na busca pela construção de máquinas inteligentes, um modelo que ocorre naturalmente é o do cérebro humano. Em nosso
    cotidiano, realizamos diversas tarefas que requerem atenção a diferentes eventos ao mesmo tempo e o processamento de
    informações variadas, de forma a tomarmos ações adequadas. Tarefas consideradas simples, como pegar um objeto ou mesmo
    caminhar, envolvem diversas habilidades, tais como memória, aprendizado e coordenação física. A complexidade dessas ações é
    demonstrada pela dificuldade de ensinar robôs a fazê-las. 
    
    A partir dessas motivações, o desenvolvimento das RNAs tomou como inspiração a estrutura e funcionamento do sistema nervoso,
    com o objetivo de simular a capacidade de aprendizado do cérebro humano na aquisição de conhecimento. Atualmente, as RNAs
    figuram entre os modelos mais eficientes e utilizados para a solução de vários problemas. Elas são empregadas em diversas
    aplicações e têm sido alvo de grande foco de pesquisa e investimento.
    
    # O SISTEMA NERVOSO
        O sistema nervoso, do qual faz parte o cérebro, é um conjunto complexo de células que determinam o comportamento dos seres
        vivos. A unidade fundamental do sistema nervoso é o neurônio, que se diferencia das outras células por demonstrar 
        excitabilidade; isso é, resposta a estímulos internos e externos.
        
        O neurônio é formado por dendritos, corpo celular e axônio. Os dendritos recebem estímulos nervosos provenientes de outros
        neurônios ou do ambiente, e envia para o corpo celular (também chamado de soma). Dependendo do estímulo, o neurônio envia
        outro impulso, que é enviado para o axônio, este que envia esse estímulo para outro lugar mais distante (por exemplo,
        outro neurônio ou outra parte do corpo).
        
        No cérebor humano, existem 10 a 500 bilhões de neurônios. Eles estão organizados em aproximadamente 1000 módulos, cada um
        com 500 redes neurais. Cada neurônio pode estar conectado a centenas ou milhares de outros neurônios, e essas redes
        trabalham de forma massivamente paralela — o que traz grande rapidez de processamento.
        
    # COMPONENTES BÁSICOS DAS RNAs
        As RNAs são sistemas computacionais distribuídos compostos por unidades de processamento simples, densamente conectadas.
        Essas unidades, conhecidas como neurônios artificiais, computam funções matemáticas. As unidades são dispostas em uma
        ou mais camadas e interligadas por diversas conexões. Cada conexão normalmente tem um peso associado, que ponderam a
        entrada recebida por cada neurônio da rede. Os pesos podem ser positivos ou negativos, a depender do comportamento ser
        excitatório ou inibitório. Os pesos têm seus valores ajustados em um processo de aprendizado e codificam o conhecimento
        adquirido pela rede.
        
        Portanto, uma RNA é caracterizada por dois aspectos básicos: arquitetura e aprendizado.
        
        1 - ARQUITETURA: O neurônio é a unidade de processamento fundamental de uma RNA. Na figura 7.2 é apresentado um modelo
            simplificado de neurônio artificial. As unidades de processamento desempenham um papel muito simples. Cada terminal
            de entrada (simulando dendritos) recebe um valor. Os valores recebidos são ponderados e combinados por uma ou mais
            funções matemáticas, e a saída da função é a resposta do neurônio para a entrada. Diversas funções matemáticas (por
            exemplo, linear, limiar, sigmoidal, tangente hiperbólica, gaussiana ou linear retificada(ReLU)) podem ser utilizadas.
            
            Nauerônios podem estar dispostos em várias camadas. Quando duas ou mais camadas são utilizadas, o neurônio pode
            receber em seus terminais de entrada valores de saída de neurônios da camada anterior e/ou enviar seu valor de saída
            para terminais de entrada de neurônios da camada seguinte. Chamamos essas redes de "redes multicamadas". A camada
            final, que gera os valores de saída, é chamada de camada de saída. Em uma rede multicamadas, as conexões entre os
            neurônios podem assumir diversos padrões, permitindo classificá-los em:
                # Completamente Conectada: quando os neurônios estão conectados a todos os neurônios da camada anterior e/ou
                    seguinte.
                #Parcialmente Conectada: quando os neurônios estão conectados a apenas alguns dos neurônios da camada anterior e
                    /ou seguinte.
                #Localmente Conectada: são redes parcialmente conectadas, em que os neurônios conectados a um neurônio se
                    encontram em uma região bem definida.
                    
            Por fim, vale mencionar a capacidade de retroalimentação; em alguns casos, é interessante fazer com que um neurônio
            receba a saída de um neurônio da mesma camada que ele, ou então de uma camada posterior. O neurônio pode inclusive
            receber a sua própria saída em um de seus terminais de entrada. Essas redes com retropropagação (redes recorrentes)
            são indicadas para aplicações em que é necessário processar informações sequenciais e na simulação de sistemas 
            dinâmicos, muito utilizada em PLNs e controle de braços robóticos. Redes sem conexões de retropropagação são mais
            comumente utilizadas na prática, e são chamadas RNAs feedfoward.
            
        2 - APRENDIZADO: Existem vários algoritmos para ajuste dos parâmetros de uma RNA, e por isso dificilmente um será feito
            do absoluto zero. Por ajuste dos parâmetros, entende-se a definição dos pesos às conexões de rede que fazem com que
            o modelo obtenha melhor desempenho, geralmente medido pela acurácia preditiva. Esses algoritmos, chamados de
            algoritmos de treinamento, são formados por um conjunto de regras bem definidos e seguem os paradigmas do aprendizado
            supervisionado, não-supervisionado e por reforço, podendo ser divididos em 4 grupos:
            
                # Correções de Erro: Geralmente utilizados em aprendizado supervisionado, procuram ajustar os pesos da RNA
                    de forma a corrigir os erros cometidos pela rede.
                
                # Hebbiano: frequentemente usados em aprendizado não-supervisionado, são baseados na regra de Hebb, que diz que,
                    se dois neurônios estão simultâneamente ativos, a conexão entre eles deve ser reforçada.
                    
                # Competitivo: utilizados em aprendizado não-supervisionado, promovem uma competição entre neurônios para definir
                    qual ou quais devem ter seus pesos ajustados. Os neurônios que vencem a competição, em geral, são os que
                    respondem mais fortemente ao objeto apresentado aos seus terminais de entrada.
                    
                # Termodinâmico(Boltzmann): algoritmos ectocásticos baseados em princípios observados na metalurgia.
                
#### REDES PERCEPTRON E ADALINE
    A primeira RNA a ser implementada foi a rede Perceptron. Essa rede utiliza o modelo McCulloch-Pitts como neurônio, e foi ela
    quem introduziu o processo de treinamento de RNAs. A rede é simples, com apenas uma camada de neurônios, mas detém boa
    acurácia preditiva em divevrsos problemas de classificação. A rede é treinada por um algoritmo supervisionado de correção
    de erro e usa a função de ativação do tipo limiar.
    
    Para um objeto xi, os pesos são ajustados de acordo com a equação:
                w^j(t+1) = w^j(t) + n*((x_i)^j)*(y_i - f(x_i))
            Onde w^j(t) é o peso da j-ésima conexão de entrada no instante de tempo t, n é uma taxa de aprendizado, (x_i)^j é o
            valor do j-ésimo atributo do vetor de entrada x_i, f(x_i) é a saída produzida pela rede no instante de tempo t e y_i
            é a saída desejada para a rede (rótulo de x_i).
    
    Alguns anos depois de propor a rede Perceptron, Rosenblatt (1958) provou o teorema de convergência da rede, que diz que,
    se é possível classificar um conjunto de entradas linearmente, uma rede Perceptron fará a classificação.
    
    Outra RNA com apenas uma camada que surgiu na mesma época da rede Perceptron é a rede Adaline. As principais diferenças entre
    as duas redes é que a rede Adaline usa uma função de ativação linear e, assim, leva a magnitude do erro em consideração na
    hora de ajustar os pesos da rede. Para isso, a Adaline usa uma regra de ajuste denominada Regra Delta, proposta por Widrow e
    Hoff, cuja fórmula é muito parecida com a do Perceptron. A diferença está na forma como o valor da saída produzida, f(x_i),
    é definido, sendo contínuo no caso das redes Adaline. Assim, ambas as redes são comumente usadas em problemas supervisionados
    de regressão. Em problemas de classificação, as saídas dos neurônios devem ser discretizadas. As redes Perceptron, por outro
    lado, foram propostas para solucionar problemas de classificação.
    
    Uma limitação das redes de uma camada, como as redes Perceptron e Adaline, é que elas conseguem gerar só funções de decisão
    lineares. Vamos supor que os objetos do conjunto de dados possuam apenas dois atributos de entrada. Os objetos de duas
    de duas classes, com dois atributos cada, são linearmente separáveis se após plotar cada objeto em um espaço bidimensional,
    utilizando o valor de cada atributo para definir a posição do objeto, existe uma reta que separa os objetos de uma classe dos
    objetos da outra classe. Ou seja, essas redes não vão conseguir separar duas ou mais classes se for necessário uma forma que
    não seja uma reta para separá-las.
    
    
#### PERCEPTRON MULTICAMADAS
    Para resolver problemas não linearmente separáveis utilizando RNAs, a alternativa mais utilizada é adicionar uma ou mais
    camadas intermediárias. Segundo Cybenko (1989), uma rede com uma camada intermediária pode implementar qualquer função
    contínua. A utilização de duas camadas intermediárias permite a aproximação de qualquer função, desde que sejam utilizados
    neurônios suficientes.
    
    As redes do tipo Perceptron multicamadas (multilayer Perceptron, MLP) apresentam uma ou mais camadas intermediárias de
    neurônios e uma camada de saída. A arquitetura mais comum para uma rede MLP é a completamente conectada, de forma que os
    neurônios de uma camada l estão conectados a todos os neurônios da camada l+1. Caso a camada l seja a primeira camada
    intermediária, cada um de seus neurônios estará conectado a todos os atributos de entrada.
    
    As redes multicamadas utilizam nas camadas intermediárias funções de ativação não lineares, como a sigmoidal. Uma rede com
    camadas intermediárias que usam funções de ativação lineares pode ser substituída por uma rede de uma só camada.
    
    Em uma MLP, cada neurônio realiza uma função específica. A função implementada por um neurônio de uma dada camada é uma
    combinação das funções realizadas pelos neurônios da camada anterior que estão conectados a ele. À medida que o processamento
    avança de uma camada intermediária para a camada seguinte, o processamento e a função correspondente se torna mais complexo.
        # Na primeira camada, cada neurônio aprende uma função que define um hiperplano, o qual divide o espaço de entrada em 
            duas partes.
        # Cada neurônio da camada seguinte combina um grupo de hiperplanos definidos pelos neurônios da camada anterior,
            formando regiões convexas.
        # Os neurônios da camada seguinte combinam um subconjunto das regiões convexas em regiões de formato arbitrário.
        
    # ALGORITMO BACK-PROPAGATION
        Uma rede multicamadas funciona em dois ciclos principais: o foward-pass e o backward pass.
            1 - Foward Pass: Os dados entram na camada de entrada, e passam pelas conexões, cada qual contém um peso(w) e onde 
                cada neurônio contém um viés(b). O neurônio soma os sinais, aplicando uma função de ativação (como a ReMU ou 
                sigmoide) e passando o resultado para a próxima camada. Ao término, a rede gera uma previsão, e faz o cálculo do 
                erro (Loss), onde é comparada a previsão da rede com o valor real esperado usando uma função de perda (por exemplo,
                o erro quadrático médio).
            
            2 - Backward Pass: Uma vez que saibamos o tamanho do erro, precisamos saber de quem é a culpa. O erro então viaja da
                da saída de volta para a entrada, e o algoritmo calcula quanto cada peso e cada viés contribuiu pro erro total.
                Isso é feito utilizando a Regra da Cadeia, uma vez que é possível calcular a derivada do erro em relação a cada
                parâmetro.
                
        Note que o back-propagation em si não altera os pesos; apenas calcula as "instruções de ajuste" chamadas de gradiente.
        Quem altera os pesos é o Otimizador (Optimizer). Geralmente, o otimizador utilizado é o Gradiente Descendente (ou uma de
        suas variações), mas também existe o ADAM(Estimativa de Momento Adaptável), Rprop, Quickprop, Newton, Levenberg-Marquardt
        e Adagrad.
        
        A fórmula de atualização é:
                    W(t+1) = W(t) - (n * Gradiente)
                Onde W(t+1) é o peso novo, W(t) é o peso antigo, n é a taxa de aprendizado(que é utilizada como o tamanho do passo,
                ou seja, se for muito grande, você passa do ponto, e se for muito pequeno, o treino demora uma eternidade) e o
                Gradiente indica a direção e a inclinação da subida. Como queremos diminuir o erro, caminhamos no sentido oposto
                (e, por isso, há um sinal negativo).
                
        Existem diversas variações do GD, dentre elas:
            1 - Gradiente Descendente Estocástico(GSD): Atualiza o peso após um único exemplo. É rápido e errático, o que ajuda a
                escapar de mínimos locais, mas nunca "sossega" no ponto ideal.
            2 - Batch Gradient Descent: Calcula o erro de todo o conjunto de dados antes de atualizar os pesos. É estável, mas
                muito lento e pesado para a memória.
            3 - Mini-batch Gradient Descent: O meio-termo ideal. Divide os dados em pequenos grupos (ex: 32 ou 64 exemplos). É o
                padrão usado na indústria hoje.
                
        O objetivo do treinamento é a convergência: o ponto onde o erro é o mínimo possível e os pesos param de mudar de forma
        significante. Este ponto é o mínimo global. No entanto, a superfície de erro do back-propagation apresenta, também,
        mínimos locais, os quais é possível que o treinamento fique preso, fazendo com que a rede tenha baixa acurácia preditiva.
        
        A depender da distribuição estatística dos objetos, a rede pode convergir para um mínimo local ou demorar para encontrar
        uma solução adequada. Esse último é um ponto negativo do algoritmo back-propagation; a sua lentidão para a convergência de
        um bom conjunto de pesos e a sua queda de desempenho quando usado em grandes conjuntos de dados e problemas complexos.
        Mesmo para problemas simples, pode ser necessário apresentar o conjunto de treinamento centenas ou milhares de vezes, o
        que frequentemente limita seu uso ao treinamento de pequenas redes, com no máximo poucos milhares de pesos. Embora alguns
        problemas práticos possam ser tratados com redes desse tamanho, há problemas reais que demandam redes maiores e mais
        complexas.
        
    # PROJETO DE ARQUITETURA DE UMA RNA
        O número adequado de neurônios na camada intermediária de uma RNA depende de vários fatores, como:
            -Número de exemplos de treinamento
            -Quantidade de ruído presente nos exemplos
            -Complexidade da função a ser aprendida
            -Distribuição estatística dos dados de treinamento
        O primeiro passo para que as RNAs possam induzir um modelo para um conjunto de dados é a definição da arquitetura, que
        engloba a escolha das funções de ativação e a topologia da rede (número de camadas e número de neurônios e, para redes
        multicamadas, qual o padrão de conexões e se existem conexões de retropropagação). A escolha de arquitetura, assim sendo,
        não é tarefa simples, e normalmente ocorre por tentativa e erro, onde diferentes configurações são utilizadas até achar
        uma que se adeque melhor ao projeto. Geralmente, a arquitetura é definida por um processo de busca, que pode ser
        realizada por diferentes abordagens. A seguir, as mais utilizadas:
            1 - Empírica: Consiste na realização de uma busca cega no espaço de possíveis arquiteturas. Assim, diversas RNAs são 
                utilizadas até achar uma cuja acurácia preditiva seja adequada. Apesar de ser a mais utilizada, detém um custo
                elevado de tempo e esforço.
                
            2 - Meta-Heurística: Essa abordagem gera um conjunto de variações de RNAs e combina as características que apresentam
                melhores resultados, gerando um novo conjunto de RNAs. Essa técnica envolve o uso de meta-heurísticas, geralmente
                Algoritmos Genéticos, para realizar uma busca global por RNAs eficientes. Como um grande número de RNAs diferentes
                precisa ser treinado, essa abordagem possui um custo computacional elevado.
                
            3 - Poda(ou Pruning): Nessa abordagem, uma RNA com grande número de neurônios é treinada até que se alcance a precisão
                desejada. Um algoritmo de poda é utilizado durante ou após o treinamento para remover conexões ou neurônios 
                redundantes ou irrelevantes das camadas intermediárias, esperando-se como resultado melhorar a capacidade de 
                generalização da rede.
            
            4 - Construtiva: A abordagem construtiva gradualmente insere novos neurônios e conexões em uma RNA inicialmente sem
                neurônios intermediários, procurando melhorar seu desempenho diante do problema em questão.

#### REDES PROFUNDAS E APRENDIZADO PROFUNDO
    A grande quantidade de dados disponíveis e gerados de maneira crescente, associada ao desenvolvimento de processadores mais
    rápidos, como as unidades de processamento gráfico (GPUs), minimizaram o problema de superajustes para RNAs com muitas
    camadas. Mais importante, permitiram que as redes profundas (DNs, do inglês, Deep Networks) treinadas por algoritmos de
    Aprendizado Profundo (DL, do inglês, Deep Learning) apresentassem desempenho preditivo significativamente superior ao obtido
    por outros algoritmos de AM, principalmente em processamento de imagens, processamento de voz e processamento de língua
    natural. Por isso, diversas empresas apostam no uso de DN em várias aplicações.
    
    Inicialmente, é importante caracterizar o que é uma DN: uma RNA MLP é denominada DN se possui ao menos duas camadas
    intermediárias. Do contrário, é chamada de rede rasa(SN, Shallow Network).
    
    Em aplicações clássicas, as RNAs Cognitron e Neocognitron, que simulavam o córtex visual, foram usadas com sucesso para o
    reconhecimento de imagens de caracteres. O desempenho de algoritmos de AM depende de quão bem os objetos do conjunto de
    treinamento representam o problema a ser tratado. Em várias aplicações, ao invés de utilizar as imagens originais, são
    extraídas características de cada imagem que possam representar bem as informações nela presentes. Quando melhor as
    características representarem a imagem original, maior a chance da imagem original ser corretamente classificada. Por isso,
    o estudo e desenvolvimento de extratores de características, chamado de engenharia de atributos, é uma importante área de
    pesquisa.
    
    No entanto, extrair características pode ter um custo elevado. Uma das principais vantagens das DNs é que elas podem aprender
    a extrair características relevantes a partir dos dados. Assim, elas conseguem extrair características gerais para determinado
    domínio de aplicação, por exemplo, imagens. Existem, aliás, DNs voltadas exclusivamente para a extração de características(a
    exemplo, Word2Vec, para extrair vetores de características numéricas de palavras, ou DNA2Vec, para extrair vetores de 
    características numéricas de sequências de RNA).
    
    As DNS são geralmente compostas por 2 módulos: o primeiro engloba as primeiras camadas e os pesos em geral são definidos
    usando aprendizado não supervisionado. As camadas no primeiro estágio são treinadas para extrair características relevantes
    dos dados originais. Para o segundo estágio, que inclui as camadas seguintes, é geralmente utilizada uma rede completamente
    conectada treinada com um algoritmo de aprendizado supervisionado como, por exemplo, o algoritmo back-propagation.
    
    As primeiras camadas da rede extraem características simples dos dados brutos. Camadas sucessivas usam as características
    extraídas pelas camadas anteriores para extrair características mais sofisticadas e complexas. Assim, o pré-treinamento cria
    níveis de representação cada vez mais complexos. Cada nível de representação pode ser visto como um módulo ou camada que
    executa um processamento simples e não linear.
    
    Existem várias abordagens populares para DL. As principais arquiteturas e abordagens de DNs que se tornaram tanto na pesquisa
    quanto nas aplicações são:
    
#### REDES NEURAIS CONVOLUCIONAIS
    Uma das abordagens mais populares de DNs pré-treinadas são as redes neurais convolucionais (CNNs, Convolutional Neural
    Networks), também conhecidas como ConvNets. Grande parte da fama conquistada por DL se deve ao excelente desempenho no
    reconhecimento de imagens. As CNNs imitam o processamento de imagens realizado pelo cérebro onde, a partir de características
    muito simples, linhas e curvas, são extraídos padrões de crescente complexidade.
    
    Uma CNN é composta por diferentes tipos de camadas organizadas em dois estágios de processamento:
        1 - Primeiro Estágio: Possui uma sequência de pares de camadas: uma camada convolucional, que extrai mapas de
            características de entrada usando filtros; seguida por uma camada de pool, que mantém apenas as informações relevantes 
            dos mapas de características. Como resultado, a representação dos dados de entrada se tornam cada vez mais abstrata. 
            Quanto mais camadas são usadas, mais complexas são as representações extraídas.
            
        2 - Segundo Estágio: Geralmente é uma rede MLP convencional. Assim, as características extraídas pela camada de estágio
            inicial podem ser usadas como atributos preditivos pela rede MLP. 
    
    Devido à sua eficácia em lidar com dados não estruturados, as CNNs são amplamente utilizadas em muitas áreas:
        # Reconhecimento de Imagens: Modelos como AlexNet, VGGNet e ResNet revolucionaram competições de visão computacional, 
            superando modelos tradicionais e redes neurais totalmente conectadas. Eles classificam imagens em categorias e também
            identificam objetos em uma imagem. É uma das aplicações mais conhecidas das CNNs.
        # Detecção de Objetos: Além de classificar uma imagem, o modelo também identifica a posição de objetos dentro da imagem.
            Algoritmos como YOLO e Faster R-CNN são exemplos de CNNs voltadas para essa tarefa, frequentemente utilizados em
            sistemas de vigilância, carros autônomos e em smartphones para detecção de rostos.
        # Segmentação de Imagens: Cada pixel de uma imagem é classificado em uma categoria específica. Isso é especialmente útil
            em áres como medicina, em que modelos são usados para segmentar tecidos ou identificar tumores em imagens de 
            ressonância magnética. Algoritmos como U-Net são projetados especificamente para isso.
        # Processamento de Áudio: Nos algoritmos de reconhecimento de fala e análise de sinais, o áudio é convertido em um
            espectrograma, permitindo que a CNN aplique convoluções igual faz com imagens.
            
    As principais vantagens das CNNs podem ser resumidas em:
        # Eficiência na extração de características: Elas são muito boas na extração de padrões locais em imagens e outros dados
            não estruturados, identificando bordas, texturas e formas de maneira hierárquica.
        # Redução de parâmetros: A operação de convolução permite o compartilhamento de pesos, reduzindo o número de parâmetros
            comparado a uma rede neural tradicional, o que diminui a complexidade e risco de overfitting.
        # Aprendizado hierárquico: A estrutura das CNNs permite que camadas mais profundas aprendam características mais
            complexas e abstratas a partir de características básicas aprendidas nas camadas iniciais.
    
    Por sua vez, as principais limitações conhecidas são:
        # Alto custo computacional: Apesar de eficientes, o treinamento de grandes modelos pode ser computacionalmente custoso,
            especialmente para dados de alta resolução e grande volume de dados. O uso de GPUs é frequentemente necessário para
            acelerar o treinamento.
        # Sensibilidade a dados de entrada: CNNs podem ser sensíveis a pequenas mudanças nos dados de entrada, como rotações ou
            mudanças na escala de uma imagem. Estratégia como o aumento de dados (data augmentation) são frequentemente usadas
            para lidar com essas variações.
    
#### REDES NEURAIS RECORRENTES:
    As Redes Neurais Recorrentes (RNN, Recurrent Neural Networks) são uma classe especial de redes neurais projetadas para lidar
    com dados sequenciais, em que a ordem das entradas é extremamente importante. Elas são particularmente úteis quando o contexto
    anterior pode influenciar previsões futuras, como no caso de texto, áudio ou séries temporais.
    
    Diferentemente das redes neurais feedfoward, em que os dados fluem em uma única direção, as RNNs têm conexões cíclicas que
    permitem que informações do passado sejam "lembradas" e possam influenciar o processamento atual.
    A principal característica que diferencia a arquitetura das RNNs de outras redes é a presença de uma unidade de "memória".
    Essa memória é implementada por meio de um estado oculto (hidden state) que armazena informações de etapas anteriores. De
    maneira simplista, cada vez que uma nova entrada é processada, a RNN atualiza seu estado oculto com base na entrada atual e
    no estado oculto anterior. Isso permite que a rede mantenha uma espécie de "memória" ao longo de uma sequência. Uma
    arquitetura típica de uma RNN é composta pelas seguintes camadas:
        # Camada de entrada: Recebe uma sequência de dados (por exemplo, um texto, uma sequência de notas musicais ou uma série
            temporal com preços de ações).
        # Camada oculta: Contém as unidades recorrentes, que atualizam seu estado com base na entrada atual e no estado anterior.
            Isso cria uma conexão cíclica na rede, permitindo que informações anteriores fluam recorrentemente.
        # Camada de saída: Produz a saída desejada para cada ponto da sequência ou para a sequência como um todo.
        
        A recorrência de uma RNN pode ser representada por:
                    h(t) = f(w_(xh)*x_t + W_(hh) * h_(t-1) + b_h)
                    
        Enquanto a saída da rede no tempo t pode ser computada por:
                    y_t = f(W_(yh)*h_t + b_y)
                    
        Em que:
            - h_t é o estado oculto no tempo t;
            - x_t é a entrada no tempo t;
            - y_t é a saída no tempo t;
            - W_(xh), W_(hh) e W_(yh) são os pesos que conectam a entrada, o estado oculto anterior à camada atual e a saída;
            - b_h e b_y são os viéses;
            - f é a função de ativação, que pode ser uma tangente hiperbólica(tanh) ou ReLU, por exemplo
        
        As RNNs são eficientes pra trabalhar com dados sequenciais. Elas são capazes de capturar as dependências temporais e de
        processar dados de comprimento variável, o que as torna flexíveis para uma ampla gama de problemas, como PLN, reconhecer
        fala, gerar texto, prever séries temporais e gerar música. Contudo, as RNNs tradicionais costumam enfrentar problemas
        quando lidam com dependências de longo prazo, ou seja, quando o modelo precisa "lembrar" de algo distante na sequência.
        Isso ocorre devido ao problema do gradiente desaparecido (gradient vanishing), em que gradientes usados para atualizar
        os pesos da rede tornam-se muito pequenos, dificultando o aprendizado de relações que ocorrem em tempos longos.
        
        Para superar essas limitações, variantes foram introduzidas, sendo as mais conhecidas:
            # LSTM(Long Short-Term Memory): Uma forma aprimorada de RNNs que incluem um mecanismo de portas que controlam o fluxo
                de informações através do tempo, permitindo que a rede "selecione" quais informações manter e quais descartar, o
                que ajuda a mitigar o problema do gradiente desaparecido e melhora a capacidade da rede de aprender dependências
                de longo prazo.
            # GRU (Gated Recurrent Unit): Também usam portas, mas têm uma arquitetura mais simples e menos parâmetros. Elas
                oferecem desempenho comparável em muitas tarefas e costumam ser mais eficientes em termos de memória.
                
                
'''
