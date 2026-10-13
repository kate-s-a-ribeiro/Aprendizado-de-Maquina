# Dados podem ser classificados de diversas formas
# Inicialmente, eles assumem valores Qualitativos(quando tem qualidades) e Quantitativos(quando tem quantidades)
'''

Atributos Quantitativos(Numéricos): são estritamente numéricos, e podem ser Contínuos ou Discretos
    1- Atributos Quantit. Contínuos: Podem assumir um número infinito de valores. Em geral, são resultados de medidas.
        Frequentemente, esses valores são representados por números reais.
            Ex: Peso, tamanho e distância
    2- Atributos Quantit. Discretos: Contêm um número finito ou infinito contável de valores. Um caso especial dos
        atributos discretos são os atributos binários (ou booleanos), que apresentam apenas dois valores,
        como 0/1, sim/não, presença/ausência, verdadeiro/falso...
            Ex: Idade, número de internações.
            
Atributos Numéricos podem assumir valores binários, inteiros ou reais. Por outro lado, atributos qualitativos
são, geralmente, representados por um número finito de símbolos ou nomes. Entretanto, algumas vezes, atributos
categóricos são representado por números, mas não faz sentido utilizar operações aritméticas neles.
Por exemplo, qual o sentido de calcular a média do número de identificação de um paciente?
'''

# A escala define as operações que podem ser realizadas
# Em relação à escala, os atributos podem ser Nominais, Ordinais, Intervalares e Racionais
# Nominais e Ordinais são qualitativos, e Intervalares e Racionais são quantitativos

'''
Nominal: Na escala nominal, os valores são apenas nomes diferentes, carregando a menor quantidade de informação possível.
         Assim, as operações mais usadas são de desigualdade ou igualdade.
         Por exemplo, se um atributo representa os continentes do planeta, é possível apenas ver se são iguais ou não.
            Ex: Nome do paciente, RG, CPF, N° da conta bancária, CEP, cores, sexo...
            
Ordinal: Refletem também uma ordem de categorias, permitindo usar >=, =<, <, >. Por exemplo, quando há valores pequeno,
         médio e grande, dá pra dizer se um valor é igual, maior ou menor que outro.
            Ex: Hierarquia militar e avaliações qualitativas de temperatura, como frio, morno e quente.
            
Intervalar: São números que variam em um intervalo, permitindo definir tanto a ordem como a diferença entre dois valores.
             A diferença em magnitude significa a distância que separa dois valores num intervalo de possíveis valores.
             O valor zero não tem o mesmo valor que 0 usado em operações aritméticas. Por exemplo, se a temperatura de uma
             mesma cidade varia entre 26 e 34 no verão e entre 13 e 21 no inverno, temos que a variação é 8 graus independente
             da estação. No entanto, 90°C é diferente de 90 graus Fahrenheit, apesar de ambos medirem temperatura, assim como
             não faz sentido dizer que o inverno é duas vezes menos quente que no verão. Isso ocorre porque o ponto zero é
             um valor arbitrário — uma característica de valores intervalares.
                    Ex: Duração de um evento em minutos e datas em um calendário.
                    
Racional: São os que carregam mais informações. Números tem um valor absoluto aqui. Por exemplo, no número de vezes que uma
          pessoa foi no hospital, o ponto zero está associado a nenhuma visita. Se um paciente teve duas internações e o outro
          teve oito, é correto dizer que um teve quatro vezes mais internações que o outro. Ou seja, faz sentido usar razão
          entre esses dois valores.
                    Ex: Tamanho, distância e valores financeiros, como salário ou saldo na conta-corrente.

'''

# Explorar esses dados podem nos trazer muitas informações importantes
# Podemos extrair medidas usando uma área da estatística denominada estatística descritiva
# Ela resume de forma quantitativa as principais características de um conjunto de dados, que podem ser calculadas rapidamente
# A exemplo, calcular a idade média dos pacientes e porcentagem de pacientes de determinado sexo

# As medidas da estatística descritiva permite capturar informações como:
'''
    -Frequência: Mede a proporção de vezes que um atributo assume um dado valor em um conjunto de dados. Pode ser aplicada
                tanto a valores quantitativos quanto qualitativos. Ex: % de pacientes com febre.
    -Localização ou tendência central (por exemplo, a média)
    -Dispersão ou espalhamento (por exemplo, o desvio padrão)
    -Distribuição ou formato

As demais medidas dependem muito de estarmos falando de dados univariados (um atributo) ou multivariados (dois ou mais).

DADOS UNIVARIADOS
    Medidas de Localidade: Definem pontos de referência nos dados e variam para dados qualitativos e quantitativos.
        Para dados simbólicos, utiliza-se geralmente a MODA (valor encontrado com maior frequência).
        Para dados numéricos, usa-se muito a MÉDIA, MEDIANA e PERCENTIL.
            -MÉDIA: O valor médio desse conjunto pode ser facilmente calculado somando todos os n atributos e dividindo por n.
                 Um problema da média é a sua sensibilidade à presença de outliers (valores muito diferentes dos demais).
                 Assim, a média é uma boa representação de um meio de valores apenas se estiverem distribuidos simetricamente.
                 Um valor muito mais baixo ou muito mais alto pode afetar muito a média e precisa ser retirado.
                    -> Existe, ainda, a média winsorizada, onde substituímos os valores que passam do limite permitido substituindo
                        -os pelo limite e entao calcula-se a média. Ex: 80% de [1,2,3,3,6,8,9,9,10] vira [2,2,3,3,6,8,9,9,9]
            -MEDIANA: Menos sensível a outliers. Ordena-se todos os n valores e, se for par, faz-se a média dos dois valores
                 no meio (1/2 * (xr + xr+1), onde n = 2r). Se for ímpar, pega o valor no meio (xr+1, onde n = 2r+1).
            -PERCENTIL E QUARTIS: Enquanto a mediana ordena e pega os valores no meio, os quartis dividem os valores em quartos.
                 Assim, o 1° quartil de uma sequência é o valor maior que 25% dos demais valores abaixo dele. Esse é o
                 25 percentil. O 2° quartil é a mediana, e o 3° quartil é o valor que tem 75% dos demais valores abaixo dele.
                 Calcula-se com percentil = [k/100 * n], onde k é o percentil desejado e n o número de atributos. Arredonda-se
                 para cima.
    
    Medidas de Espalhamento: As medidas de espalhamento medem a dispersão de um conjunto de valores. Assim, permite ver se os
        valores estão amplamente espalhados ou concentrados em torno de um valor, por exemplo, a média. As medidas de
        espalhamento mais comuns são o INTERVALO, VARIÂNCIA e DESVIO-PADRÃO.
            -INTERVALO: Mostra o espalhamento máximo do conjunto, isso é, a distância entre o maior valor e o menor valor.
                  Se a maioria dos valores forem próximos uns dos outros, com um pequeno número de valores extremos, não é
                  muito utilizado o Intervalo.
            -VARIÂNCIA: A mais utilizada, é medida ao quadrado. Calcular a variância significa calcular a média e a distância
                  de cada valor à média. Elevamos essa distância ao quadrado e então fazemos a média desses quadrados.
                  Assim como a média, a variância é muito afetada pela presença de outliers. 
                    -> Podemos, no cálculo da média, utilizar um denominador n-1 ao invés de n. Isso é chamado de Correção de Bessel.
            -DESVIO-PADRÃO: É a raiz-quadrada da variância(trazendo os dados para a medida normal), mais facilmente interpretada.
            
    Medidas de Distribuição: As medidas definidas em torno da média de um conjunto de valores, como a média e desvio-padrão,
        são (normalmente) instâncias de uma medida denominada momento. A equação do momento é difícil de dizer (pesquise), mas:
            1° Momento(k = 1): Tem-se a média.
            2° Momento(k = 2): Tem-se a variância.
            3° Momento(k = 3): Tem-se a Obliquidade/Assimetria, mostrando se os dados se concentram à esquerda ou à direita.
                                -Se Obliquidade = 0, a distribuição é aproximadamente simétrica.
                                -Se obliquidade > 0, se concentra do lado esquerdo.
                                -Se Obliquidade < 0, se concentra no lado direito.
            4° Momento(k = 4): Tem-se a Curtose, que calcula o achatamento da distribuição.
                                -Se Curtose = 0, tem o achatamento de uma distribuição normal.
                                -Se Curtose > 0, tem uma dsitribuição mais alta e centrada que a normal.
                                -Se Curtose < 0, tem uma distribuição mais achatada que a normal.
            -> O momento pode ser padronizado ao redor da média, fazendo (xi - média de x) para cada xi.
                                
                                
                                
DADOS MULTIVARIADOS
São dados que possuem mais de um atributo de entrada. Nesses casos, as medidas de localidade são obtidas calculando a medida
de cada atributo separadamente. Por exemplo, a média para um conjunto de objetos com D atributos é feita fazendo a média de cada
um dos D atributos.

As medidas de espalhamento também podem ser calculadas para cada atributo independentemente dos demais utilizando qualquer
medida de espalhamento.

Dados multivariados permitem análises da relação entre dois ou mais atributos. Por exemplo, para atributos quantitativos, o
espalhamento de um conjunto de dados é mais bem capturado por uma matriz de COVARIÂNCIA, em que cada elemento é a COVARIÂNCIA
entre dois atributos. Cada elemento cov_ij de uma matriz de COVARIÂNCIA mede a COVARIÂNCIA entre os atributos x_i e y_j.

Para calcularmos a COVARIÂNCIA:
    1 - Fazemos uma lista de valores X e de valores Y.
    2 - Encontramos a média dos valores X e a média dos valores Y. Para fim de entendimento, chamaremos de 'X e 'Y;
    3 - Para cada par XY, calculamos (X_i - 'X)*(Y_i - 'Y), para cada um dos i
    4 - Somamos tudo
    5 - Dividimos por n - 1 (ou seja, dividimos a soma pelo número de pares - 1)
    
Se a COVARIÂNCIA for:
    -COVARIÂNCIA(x_i, y_j) > 0, as variáveis aumentam e diminuem juntas.
    -COVARIÂNCIA(x_i, y_j) < 0, uma variável aumenta quando a outra diminui e vice-versa
    -COVARIÂNCIA(x_i, y_j) = 0, não há relação linear aparente
    
Note, no entanto, que dois atributos cujos valores variam muito podem apresentar COVARIÂNCIA maior que dois atributos mais 
semelhantes entre si, mas de menor variação de valores. Por isso, não dá pra avaliar o relacionamento entre dois atributos
observando somente a COVARIÂNCIA. E aí entra outra medida:

CORRELAÇÃO:
A medida de CORRELAÇÃO retira esse problema pois retira a influência da variação dos valores. Portanto, ela é uma indicação
mais clara da força da relação linear entre dois atributos. Assim, é mais utilizada em dados multivariados. Assim como a
covariância, a CORRELAÇÃO é uma matriz que apresenta a CORRELAÇÃO entre o atributo x_i e y_j.

Para se calcular a CORRELAÇÃO:
    1 - Primeiro calcula-se a COVARIÂNCIA.
    2 - Depois, calcula-se o desvio padrão de cada atributo x_i e de cada atributo y_j. (chamemos de s_xi e s_yj)
    3 - Para cada x_i e y_j, divide-se a COVARIÂNCIA(x_i, y_j) pelo produto s_xi * s_yj.
    4 - Assim, nós temos a CORRELAÇÃO(x_i, y_j).
            
'''


'''
    Boxplots = representa os intervalos entre os quartis. No entanto, o intervalo por ser muito grande por causa de outliers.
        Nesse caso, podemos fazer com que a tolerância seja = 1.5 * IQR, onde IQR = Q3 - Q1. Assim, o Limite Inferior vai ser
        Q1 - tolerância e o Limite Superior vai ser Q3 + tolerância.
            Verificar se máximo - Q3(Q1-mínimo) > tolerância.
'''
