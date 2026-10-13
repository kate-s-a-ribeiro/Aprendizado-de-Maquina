# matplotlib
# básico de plotar gráficos e usar matemática
import matplotlib.pyplot as plt # poderíamos usar "from matplotlib import pyplot as plt"
import numpy as np

# como um gráfico é com base no tempo, precisamos ter um t
# o linspace permite criar um array de valores de um determinado valor inicial(t = 0) até um final(t = 5)
# e quantas partições vai ter esse array
t = np.linspace(0, 5, 10) # vai dividir o intervalo de 0s a 5s em 10 pedaços

print(t) # vai printar os 10 períodos

# se quisermos ver o valor do cosseno de t para cada um dos 10 valores de t entre 0 e 5, fazemos simplesmente:
y = np.cos(t)
print(y)

# agora, vamos plotar um gráfico do cosseno:
#plt.plot(t, y) # gerou um gráfico do cosseno em relação ao tempo
#plt.show()    # e agora vai mostrar esse gráfico

# opa, o cosseno não completou um ciclo. Vamos fazer ele completar um ciclo
# para que a curva fique menos "quadrada", vamos aumentar a quantidade de "pedaços":

t = np.linspace(0, 2*np.pi, 1000)
y = np.cos(t)
#plt.plot(t, y)
#plt.show()

# podemos, ainda, aumentar a quantidade de períodos. Se t = 1 período, então podemos fazer 4t, 5t ou 100t
# Além disso, o matplotlib nos permite dar um título ao gráfico, além de legendas pro eixo x e y

# para salvarmos o gráfico, podemos fazer:
'''
plt.figure("Cosseno")   # o gráfico será salvo dentro dessa figura "Cosseno"
y = np.cos(4*t)
plt.plot(t, y)
plt.title("Gráfico do Cosseno")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude (cm)")
'''

# vamos criar uma figura dois:
'''
plt.figure("Seno", figsize=(5, 4))  # podemos alterar o tamanho da figura, que será 5 por 4 polegadas
y2 = np.sin(4*t)
plt.plot(t, y2)
plt.title("Gráfico do Seno")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude (cm)")
'''


#plt.show()      # vai mostrar as duas figuras


### Vamos agora fazer outros gráficos, como de exponencial:
'''
t1 = np.arange(0, 100)  # uma rede de números inteiros do 0 ao 100
y1 = np.exp(t1)   # o eixo y será a exponencial(número de euler) do eixo t
plt.figure("Gráfico", figsize=(6, 4))
plt.plot(t1, y1)
'''

# podemos, ainda, colocar mais de uma função no mesmo gráfico:
'''
t = np.linspace(0, 4*np.pi, 100)
y = np.cos(t)
y2 = np.sin(t)
plt.figure("Gráfico", figsize=(6, 4))
plt.title("Gráficos Seno e Cosseno")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.plot(t, y)
plt.plot(t, y2)
plt.show()
'''
# lembre que as especificações devem ser abaixo da figure!

# podemos também colocar mais de um gráfico numa mesma figura
# ou seja, um mesmo gráfico com vários subplots
x = np.linspace(0, 2*np.pi, 500)
c = np.cos(x)
s = np.sin(x)
 
# o subplot recebe três argumentos: a quantidade de linhas que eu quero, a quantidade de colunas que eu quero, e a posição
# podemos, em cada subplot, adicionar títulos e labels como se fosse um plot normal
'''
plt.figure("Gráficos Senoides", figsize=(7, 8))

# podemos, ainda, ajustar os subplots:
plt.subplots_adjust(
    left= 0.125,
    right=0.94,
    bottom=0.25,
    top=0.88,
    wspace=0.204,
    hspace=0.465
)

a = plt.subplot(2, 1, 1)
plt.plot(x, c)
a.grid()
a.set_title("Gráfico do Cosseno")
a.set_xlabel("Eixo de Tempo")
a.set_ylabel("Eixo de Amplitude")
b = plt.subplot(2, 1, 2)
plt.plot(x, s)
b.set_title("Gráfico do Seno")
b.set_xlabel("Eixo de Tempo")
b.set_ylabel("Eixo de Amplitude")

plt.show()
'''

# outra forma de gerar múltiplos gráficos numa só figura é:
x = np.arange(0, 5, 0.1)
y = x**5 # x elevado à quinta
y2 = x**2 # x elevado a 2
print(x)

# o interessante de fazer usando plt.subplots é poder colocar o n° de colunas e linhas diretamente ao chamar o subplots
'''fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(8,4)) # por termos duas colunas, vou ter dois ax's, logo duas variaveis
ax1.plot(x, y2)  # agora eu consigo plotar dois axs sem depender um do outro
ax2.plot(x, y)
plt.show()'''

# e se eu tiver duas colunas e duas linhas:
'''fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,4))

# eu posso ajustar cada subplot normalmente:
plt.subplots_adjust(
    left= 0.125,
    right=0.94,
    bottom=0.25,
    top=0.88,
    wspace=0.204,
    hspace=0.465
)
ax1.plot(x, y2) 
ax1.set_title("Elevado ao quadrado")
ax1.set_xlabel("Tempo")
ax1.set_ylabel("Aplitude")
ax1.grid(True)

ax2.plot(x, y)
ax3.plot(x, y2)
ax4.plot(x, y)
plt.show()'''

# Outra forma de utilizar os subplots é:
'''fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6)) # salvei quatro axis dentro de uma variável só
plt.subplots_adjust(
    left= 0.125,
    right=0.94,
    bottom=0.25,
    top=0.88,
    wspace=0.204,
    hspace=0.846
)

# acesso o axis usando [], o primeiro número sendo da linha e o segundo da coluna
axes[0, 0].plot(x, y)  # note que, se fosse só dois axis (1 linha e 2 colunas ou o contrário), não precisaria de dois números
axes[0, 0].set_title("Função de Segundo Grau $x^2$")
axes[0, 0].set_xlabel("Tempo")
axes[0, 0].set_ylabel("Amplitude")

axes[0, 1].plot(x, y2)
axes[0, 1].set_title("Função bla bla")
axes[0, 1].set_xlabel("Tempo")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].grid(True)

axes[1, 0].plot(x, y)
axes[1, 0].set_title("Função de Segundo Grau $x^2$")
axes[1, 0].set_xlabel("Tempo")
axes[1, 0].set_ylabel("Amplitude")

axes[1, 1].plot(x, y2)
axes[1, 1].set_title("Função bla bla")
axes[1, 1].set_xlabel("Tempo")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].grid(True)

plt.show()'''

### CUSTOMIZANDO
'''x = np.linspace(0, 16*np.pi, 100)
y = np.cos(x)

plt.figure(figsize=(8,4))
plt.plot(x, y, color="#FFD700", lw=1.0, marker="o",
         linestyle="dashdot") # podemos alterar a cor(inclusive com código rgb), estilo e espessura da linha, adicionar símbolos(pontos, x, etc) a cada partição
plt.grid(True)
plt.title("Gráfico do Cosseno")
plt.xlabel("Eixo do Tempo")
plt.ylabel("Eixo da Amplitude")
plt.show()'''

# customizando escala dos eixos e tamanho da fonte
'''
x = np.linspace(0, 4*np.pi, 500)
y = np.cos(x)

fig, axe = plt.subplots(figsize=(14, 8))
axe.plot(x, y)
axe.set_title("Gráfico do Cosseno", fontsize=16) # mudando o tamanho
axe.set_xlabel("Eixo X", fontsize=10)
axe.set_ylabel("Eixo Y", fontsize=10)

# mudando a escala dos eixos:
plt.xticks(np.arange(0, 4*np.pi+0.5, 0.5)) # vai ir do 0 a 4pi+0.5, de meio em meio
plt.yticks(np.arange(-1, 1, 0.1))       # vai ir do -1 ao 1, de 0.1 em 0.1


plt.grid()
plt.show()
'''


# mudando o estilo do matplotlib
import matplotlib.style as st
'''
x = np.linspace(0, 4*np.pi, 500)
y = np.cos(x)

plt.style.use("dark_background")

fig, axe = plt.subplots(figsize=(14, 8))
axe.plot(x, y)
axe.set_title("Gráfico do Cosseno", fontsize=16) # mudando o tamanho
axe.set_xlabel("Eixo X", fontsize=10)
axe.set_ylabel("Eixo Y", fontsize=10)

# mudando a escala dos eixos:
plt.xticks(np.arange(0, 4*np.pi+0.5, 0.5)) # vai ir do 0 a 4pi+0.5, de meio em meio
plt.yticks(np.arange(-1, 1, 0.1))       # vai ir do -1 ao 1, de 0.1 em 0.1


plt.grid()
plt.show()'''

# Identificando um ponto com linhas paralelas e marcador usando matplotlib, além de texto no gráfico
'''x = np.linspace(1, 5, 500)
y = np.log10(x)

plt.style.use("classic")
fig, axe = plt.subplots(figsize=(7, 4))
axe.plot(x, y, lw=1.2)
axe.plot([0, 2.5], [0.4, 0.4], color="gray", linestyle="--", lw=0.5)  # cria uma linha horizontal em y = 0.4 que vai de x = 0 até x = 2.5
axe.plot([2.5, 2.5], [0, 0.4], color="gray", linestyle="--", lw=0.5)  # cria uma linha vertical em x = 2.5 que vai de y = 0 até y = 0.4
axe.plot([0, 5], [0, 0.8], linestyle="--", lw=0.8)    # cria uma linha tangente ao ponto (2.5, 0.4) da função log10(x)
axe.plot(2.5, 0.4, marker="o")              # cria um pontinho no ponto (2.5, 0.4)
axe.text(2.7, 0.37, "P(2.5, 0.4)")          # cria um texto no ponto (2.7, 0.37)
axe.text(3, 0.43, "Logarítmo $y = log_{10}x$", fontsize=10, bbox=dict(facecolor="red", alpha=0.5))  # cria um texto dentro de uma caixinha vermelha, intensidade do vermelho 0.5 de 1, tamanho 10
axe.annotate("P(2.5, 0.4)",
             xy=(2.5, 0.4),
             fontsize=16,
             xytext=(0.5, 0.5),
             arrowprops=dict(facecolor="red",
                             alpha=1),
             color="r")      # cria uma setinha vermelha apontando para o ponto 2.5, 0.4 com o texto vermelho escrito na posição 0.5, 0.5

plt.grid()
plt.show()'''



# Agora, vamos fazer um Histograma usando matplotlib
'''idades = np.random.normal(loc=50, scale=12, size=500)   # pegando valores em uma distribuição normal, de centro 50, desvio padrão 12, com 500 amostras(valores)

plt.hist(idades, bins=20)   # bins são "caixinhas", as barras do gráfico
plt.xlabel("Idade")
plt.ylabel("Frequência")
plt.title("Distribuição de Idades de Pacientes")
plt.show()'''

# ou então:
'''pressao_sistolica = np.random.normal(120, 15, 400)

plt.hist(pressao_sistolica, bins=25)
plt.xlabel("Pressão Sistólica (mmHg)")
plt.ylabel("Número de Pacientes")
plt.title("Distribuição da Pressão Sistólica")
plt.show()'''

# também podemos fazer mapas de saúde:
# 1 - Mapa simples (coordenadas latitude/longitude)
lat = np.random.uniform(-23.7, -23.5, 200)
lon = np.random.uniform(-46.7, -46.5, 200)

plt.scatter(lon, lat)   # esse é um mapa simples, sem fundo geográfico
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Distribuição de Casos")
plt.show()

# 2 - Mapa de calor (heatmap)
dados = np.random.rand(10, 10)

plt.imshow(dados)   # transforma matriz em mapa
plt.colorbar(label="Taxa de incidência")    # mostra em escala de cores
plt.title("Mapa de Incidência de Doença")
plt.show()



