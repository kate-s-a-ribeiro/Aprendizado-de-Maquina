import pandas as pd     # para análise de dados
import numpy as np      # aqui, vamos usar um pouco de geometria analítica e álgebra linear
import os               # para acessar os arquivos certinho

# precisa colocar tudo dentro de um if pra poder colocar mais de um núcleo pra trabalhar
# se não colocar, vai demorar muito pra treinar
if __name__ == "__main__":  
    # primeiro de tudo, precisaremos acessar os arquivos com as fotos dos raio-x
    # a função os.walk entra na pasta indicada e, para cada pasta que ela encontra, ela retorna uma tupla contendo três coisas
    # dirname = o caminho da pasta atual onde ele está no momento
    # _ = uma lista com o nome das subpastas (o _ significa que não vamos usar essa variável)
    # filenames = uma lista com o nome de todos os arquivos dentro da pasta atual
    # esse for vai garantir que o código vai visitar a pasta principal ('chest_xray')
    # e todas as pastas que estiverem dentro dela
    for dirname, _, filenames in os.walk('./py/Aprendizado de Máquina/Datasets/chest_xray_pneumonia'):
        # já esse for é para garantir que, a cada pasta visitada, ele vai olhar para a lista de arquivos e iterar sobre cada um dos
        # arquivos
        for filename in filenames:
            # essa função vai juntar o caminho da pasata ('dirname') com o nome do arquivo ('filename')
            # isso significa que, se o dirname for ./Datasets/chest_xray/train, e o filename for fotodepulmao.png,
            # o resultado vai ser './Datasets/chest_xray/train/fotodepulmao.png'
            os.path.join(dirname, filename)
    #        caminho = os.path.join(dirname, filename)  # se quisermos confirmar
    #        print(caminho)

    # vamos agora definir os diretórios
    # essa parte é autoexplicativa
    diretorio_base = "./py/Aprendizado de Máquina/Datasets/chest_xray_pneumonia/chest_xray"
    diretorio_teste = os.path.join(diretorio_base, 'test')
    diretorio_treino = os.path.join(diretorio_base, 'train')
    diretorio_valor = os.path.join(diretorio_base, 'val')

    # para o processamento de imagens, vamos importar o módulo "Image" da biblioteca pillow
    # e vamos importar o Glob
    # o glob é para ajudar a encontrar caminhos que sigam determinado padrão
    # o bom do glob é que nos permite usar caracteres coringa como *
    from PIL import Image
    import glob

    # vamos usar o glob pra buscar por arquivos
    imagens_amostra_teste = glob.glob(diretorio_teste + '*/*/*.jpeg')[:]
    # o glob vai globar dentro da pasta de teste e em qualquer subpasta (/*), por qualquer arquivo que termine com .jpeg
    # o [:5] é pro glob pegar só os 5 primeiros caminhos encontrados, pra não sobrecarregar o console com infinitas imagens
    # (na realidade, nesse caso, nem precisa disso)

#    for caminho_imagem in imagens_amostra_teste: # para cada imagem no diretório de imagens
#        imagem = Image.open(caminho_imagem)      # vamos abrir a imagem
#        print(imagem.size, imagem.mode)          # e checar o tamanho da imagem(lxh) e o modo de cor da imagem(RGB ou tons de cinza)
        # dependendo do tamanho das imagens, pode ser necessário redimensioná-las
        # esse trecho todo é só pra isso


    # agora importamos as coisas do torch
    import torch
    from torchvision.transforms import v2
    from torchvision import datasets
    from torch.utils.data import DataLoader
    
    # limpar algum possível cache de uma execução anterior
    torch.cuda.empty_cache()
    
    import gc
    gc.collect()

    # primeiro de tudo, precisamos saber de algo chamado "normalização de imagem"
    # em aprendizado de máquina, normalizar uma imagem é ajustar os valores de intensidade dos pixels de uma imagem para um intervalo
    # padrão, facilitando que a convergência do modelo e deixando o treino mais rápido e/ou preciso
    # a maior parte das vezes, usamos a reescala básica (de 0 a 1), mas também podemos usar os valores da ImageNet (que é o padrão
    # da indústria), que são média = [0.485, 0.456, 0.406] e desvio padrão = [0.229, 0.224, 0.225] ou fazer a globalização global
    # simples, que é média = [0.5] e desvio padrão = [0.5] (colocando o intervalo entre -1 e +1)
    # também poderíamos calcular qual é o melhor valor para média e desvio padrão, mas é complicadinho, então vamos deixar para um
    # momento futuro

    # cada valor de média e desvio padrão tem seus prós e contras
    # para fins de facilidade, vamos estar forçando a média e desvio padrão = 0.5
    media = 0.5
    desvio_padrao = 0.5

    # agora, como a quantidade de imagens é limitada, precisamos evitar overfitting
    # além disso, fazer o que vamos fazer também vai permitir que o modelo generalize imagens em diferentes ângulos e etc
    '''transformar_treino = v2.Compose(
        [v2.Resize((384, 384)), # primeiro, vamos tornar todas as imagens em 224x224, para ser melhor pro modelo
        v2.Grayscale(num_output_channels=1),   # por default, imagefolder abre todas as imagens em RGB. Mas como as imagens são em tons de cinza, vamos forçar a imagem a ter apenas um canal (preto e branco)
        v2.RandomRotation(degrees=15),  # gira a imagem em até 10°, pro modelo entender mesmo imagens onde o pulmão tá levemente inclinado
        v2.RandomHorizontalFlip(p=0.5), # espelha a imagem horizontalmente 50% das vezes, pra ter várias imagens
        v2.RandomAffine(degrees=0, translate=(0.05, 0.05)), # move o paciente 5% pra cima e 5% pra baixo, pra simular um paciente fora do centro
        v2.ColorJitter(brightness=0.2, contrast=0.2),   # altera levemente o brilho e o contraste, para treinar ela em imagens um pouco mais claras e um pouco mais escuras
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True), # coloca a imagem em um tensor (formato que o torch entende) e coloca os valores dos pixels entre 0 e 1
        v2.Normalize(mean=[media], std=[desvio_padrao])    # normalizando os dados usando a media e desvio padrão que escolhemos
        ]
    )'''
    transformar_treino = v2.Compose([
        # Em vez de apenas Resize, usamos o RandomResizedCrop
        # Ele pega uma parte da imagem (entre 70% e 100% da original) e estica para 384x384
        # Isso tira o coração do centro e muda as bordas de lugar constantemente
        v2.RandomResizedCrop(size=(384, 384), scale=(0.7, 1.k0), antialias=True),
        
        v2.Grayscale(num_output_channels=1),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15), # Aumentamos um pouco a rotação
        
        # RandomEqualize ajuda a destacar as texturas dos pulmões que podem estar "lavadas"
        v2.RandomEqualize(p=0.2), 
        
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[media], std=[desvio_padrao])
    ])

    # vamos também alterar um pouco as imagens, dessa vez pra avaliação
    # não tem necessidade de rotacionarmos, espelharmos, movermos o paciente nem nada do tipo
    # por isso, aqui só vamos redimensionar, transformar em escalas de cinza e normalizar
    transformar_avaliacao = v2.Compose(
        [v2.Resize((384, 384)),
        v2.Grayscale(num_output_channels=1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[media], std=[desvio_padrao])]
    )

    # agora, vamos criar os datasets com base nas transformações e com base nas pastas
    data_treino = datasets.ImageFolder(root=diretorio_treino, transform=transformar_treino)
    data_teste = datasets.ImageFolder(root=diretorio_teste, transform=transformar_avaliacao)
    data_avaliacao = datasets.ImageFolder(root=diretorio_valor, transform=transformar_avaliacao)

    # agora que nós temos as datas separadas em teste e treino, podemos brincar com esses datasets
    from torch.utils.data import ConcatDataset, random_split

    # primeiramente, vamos combinar os datasets de treino e validação em um só dataset blocudo chamado "combinação"
    combinacao = ConcatDataset([data_treino, data_avaliacao])
    # então, iremos treinar esse dataset em 80% do tamanho desse blocao, e os 20% restando será para validação
    tamanho_treino = int(0.8 * len(combinacao))
    tamanho_validacao = len(combinacao) - tamanho_treino
    # e vamos dividir aleatoriamente as imagens entre teste e treino, 80% indo pro treino e 20% para validação
    data_treino, data_avaliacao = random_split(combinacao, [tamanho_treino, tamanho_validacao])

    # vamos dar uma configurada rápida nos dataloaders
    # isso não é tão importante agora, mas explicando:
    # os dataloaders vão entregar as imagens pro modelo
    # eles vão entregar 32 imagens em 32 imagens (batch_size)
    # vão embaralhar as fotos a cada remessa no treino, mas no teste e validação não irão
    # isso é porque, no treino, não podemos correr o risco do modelo aprender a ordem das fotos ao invés dos padrões da imagem
    # num_workers é a quantidade de núcleos do processador que vão preparar as imagens enquanto a GPU treina o modelo
    # é só pra acelerar o processo
    remessa = 32
    treino_loader = DataLoader(data_treino, batch_size = remessa, shuffle=True, num_workers=4)
    teste_loader = DataLoader(data_teste, batch_size= remessa, shuffle= False, num_workers=4)
    validacao_loader = DataLoader(data_avaliacao, batch_size= remessa, shuffle= False, num_workers=4)

    # agora, vamos querer visualizar as imagens que o modelo está recebendo
    import matplotlib.pyplot as plt

    # vamos criar uma função para isso
    def visualizador_de_imagens(loader, media, desvio_padrao, n=16, titulo="Amostragem de Imagens"):
        images, labels = next(iter(loader)) # transforma o dataloader em um objeto que podemos percorrer
        # assim sendo, o next vai pegar a primeira remessa de imagens
        # e o [:n] seleciona apenas as primeiras 16 (já que n=16) imagens pra mostrar no gráfico
        
        # como eu havia normalizado as imagens anteriormente, eu vou ter que "desnormalizar" pra mostrar elas
        imagens_amostragem = images * desvio_padrao + media
        imagens_amostragem = imagens_amostragem.clamp(0, 1) # para garantir que nenhum pixel fique acima de 0 ou 1
        
        # vamos só definir quantas colunas e linhas de imagens vamos querer mostrar
        colunas = 4
        linhas = n // colunas
        figura, axis = plt.subplots(linhas, colunas, figsize=(12, linhas * 3))
        
        # e agora vamos exibir as imagens
        for i, ax in enumerate(axis.flat):
            # as imagens do PyTorch tem formato (1, 224, 224), enquanto o matplotlib precisa de (224, 244)
            # o squeeze() tira esse "1" extra
            # o numpy converte pra numpy
            img = imagens_amostragem[i].squeeze().numpy()
            ax.imshow(img, cmap='gray')         # deixa as imagens em tons de cinza
            
            # definimos uma classe binária onde 0 significa que o paciente está normal, e 1 significa que ele tem pneumonia
            nomes_classe = {0: 'NORMAL', 1: 'PNEUMONIA'}
            
            # define a cor do título
            # vamos colocar verde pra normal e vermelho pra pneumonia
            ax.set_title(nomes_classe[labels[i].item()],
                        color='red' if labels[i].item() == 1 else 'green',
                        fontsize=10)
            ax.axis('off')
        
        # só alguns detalhes no título e layout
        plt.suptitle(titulo, fontsize=14, fontweight='bold')
    #    plt.tight_layout() acho que fica mais bonito sem um layout compacto
        plt.show()  # pra mostrar a imagem
        
    visualizador_de_imagens(treino_loader, media, desvio_padrao, n=16, titulo="Amostras de Treino")
    visualizador_de_imagens(validacao_loader, media, desvio_padrao, n=16, titulo="Amostras de Validacao")

    # só pra confirmar que tá tudo certo
    # o tamanho da imagem deve ser (1, 224, 224) e as classes devem ser 0 = normal, 1 = pneumonia
    '''imagem, rótulo = next(iter(treino_loader))
    print(imagem[0].size())
    print(data_treino.dataset.datasets[0].class_to_idx)'''

    # pois bem, finalmente passamos para a criação da rede neural
    # nós temos duas opções aqui:
    # podemos criar uma rede neural do zero
    # ou
    # podemos usar um modelo já pronto e alterá-lo um pouco

    # criar uma rede neural do zero é um pouco complicado logo de cara, e tem a possibilidade de ter uma acurácia menor
    # por isso, vamos usar um modelo já pronto

    # primeiramente, importamos as coisas necessárias do pytorch:
    import torch.nn as nn
    import torchvision.models as models

    # agora, definimos a arquitetura da nossa rede
    # utilizaremos uma rede profunda convolucional
    '''class modelopneumonia(nn.Module):
        # coisa normal de inicializar rede neural
        def __init__(self, *args, **kwargs):
            super().__init__()
            
            # selecionamos o nosso modelo
            # vamos testar com diferentes modelos, mas primeiro, bora ir de resnet50
            # ResNet18 com threshold = 0.5, temos mais de 87% de acurácia e 97% de recall em pneumonia, AUC-ROC máximo chegado foi 0.85
            # Com um redutor de taxa de aprendizado, o ResNet18 com o mesmo threshold tem 90.22% de acurácia e 96% de recall em pneumonia, AUC-ROC de 0.88
            self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
#            self.model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
            
            # note, no entanto, que resnet espera por imagens RGB
            # como as nossas imagens estão em tons de cinza, precisamos alterar um pouco o resnet
            
            1 - A primeira entrada (1) são os canais de entrada; se fosse RGB, seriam 3. Como é em tons de cinza, é 1
            2 - A segunda entrada são os canais de saídas, e é a quantidade de filtros que a rede vai criar. Pense na rede criando
                16 versões diferentes da sua imagem, cada uma focando em algo (bordas, sombras etc)
            3 - Kernel_Size é o tamanho da janela de visão. 7x7 é considerada grande e, assim, ela vai captar estruturas maiores
                logo de cara no raio-x. Normalmente faz-se uma janela grande só na primeira camada
            4 - Stride é o passo. Ao invés de deslizar a janela pixel em pixel, ela pula de 2 em 2, fazendo com que a imagem de saída
                tenha metade do tamanho da original. podemos colocar = 1 para que a rede analise com maior detalhes, custando mais
                para processar
            5 - Padding é a "moldura" de zeros, adicionando 3 píxeis pretos nas bordas pra rede conseguir processar os cantos da imagem
            6 - Bias é usado quando temos uma camada batchnorm logo em seguida. O ResNet tem
            
            self.model.conv1 = nn.Conv2d(
               1, 64, kernel_size=7, stride=2, padding=3, bias=False
           )
#            self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # além disso, como é uma classificação binária, temos que alterar a camada final (a camada de saída)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)   # vamos usar curva linear, mas podemos usar ReLU também
#            in_features = self.model.classifier.in_features
#            self.model.classifier = nn.Linear(in_features, 1)
 
            
        def forward(self, x):
            return self.model(x)'''

    class modelopneumonia(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
            
            # Mantemos a alteração para 1 canal
            self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Alteramos a saída para incluir um Dropout antes da camada linear
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.4), # 40% de chance de ignorar um neurônio no treino
                nn.Linear(in_features, 1)
            )
                
        def forward(self, x):
            return self.model(x)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)   # estamos usando cuda, ou seja, a GPU ao invés da CPU
    # isso é muito melhor

    # vamos atribuir o modelo, rodado na GPU, à uma variável modelo
    model = modelopneumonia().to(device)

    # pois bem, agora temos que checar se o dataset está desbalanceado
    # vamos ver quantos de cada classe tem as imagens
    from collections import Counter
    contagem_treino = Counter(data_treino.dataset.datasets[0].targets)
    print(contagem_treino)  # está desbalanceado; existem 3875 de classe 1 (pneumonia) e 1341 de classe 0 (normais)

    # vamos aumentar os pesos da classe minoritária então; assim, para cada erro que a IA cometer, vai doer mais pra ela
    # calculamos a proporção e atribuimos esse peso
    peso = torch.tensor([contagem_treino[0] / contagem_treino[1]]).to(device)
    criterio = nn.BCEWithLogitsLoss(pos_weight = peso)

    # agora, vamos fazer o otimizador para calcular os pesos!
    # podemos usar o Gradiente Descendente ou o Adam
    # vamos usar o Adam porque, bom, por que não?
    import torch.optim as optim

#    otimizador = optim.Adam(model.parameters(), lr = 0.001, weight_decay= 1e-4)
    otimizador = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    # utilizamos os parâmetros do modelo sem alterar nada, setamos a taxa de aprendizado a 0.001 e fazemos com que haja decaimento
    # nos pesos, penalizando pesos que ficarem grandes demais

    # vamos já setar essas variáveis
    perdas_treino = []
    perdas_validacao = []
    melhor_perda_validacao = float("inf")
    paciencia = 5
    contador = 0

    # podemos fazer um redutor da taxa de aprendizado, mas é um pouco mais complexo, então não iremos por agora
    # okay agora iremos
    redutor = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode='min', factor=0.3, patience=2)

    # AGORA VAMOS ENTRAR NA PARTE DE TREINAR E TESTAR O MODELO!
    # vamos pensar na lógica e depois aplicar
    # primeiro, vamos dividir em "épocas"; uma época é a leitura completa de todas as imagens do dataset
    # ou seja, se há 15 épocas, então o modelo vai ler todas as imagens 15 vezes
    # o modelo vai treinar, vai "anotar os erros dele" e, ao errar, vai olhar pra trás (backwardpass) 
    # e tentar ver qual neurônio errou pra poder alterar os pesos dele
    # após descobrir o erro, o otimizador ajusta os pesos da rede pra não errar mais na próxima vez

    # após o treino, o modelo vai testar
    # ele vai parar de de aprender e focar em responder, vai parar de calcular o erro e coisa do tipo

    # o uso do redutor da taxa de aprendizado seria para, caso o modelo não estivesse melhorando, reduzir a taxa de aprendizado dele

    # depois do teste, vamos ter que fazer condições de parada
    # vamos pegar o melhor resultado e, se o modelo passar muitas épocas sem melhorar, o código decide parar pra evitar que o modelo
    # acabe decorando as fotos (overfitting) em vez de aprender a doença

    # vamos colocar isso em código:
    j = 0
    epocas = 15
    for epoca in range(epocas):
        model.train()   # treinar o modelo
        perda_corrente = 0.0

        for images, labels in treino_loader:
            images = images.to(device)  # carregamos as imagens na GPU
            labels = labels.to(device).unsqueeze(1).float() # o modelo espera uma coluna vertical com 0s e 1s; o unsqueeze faz isso
            otimizador.zero_grad()  # MUITO IMPORTANTE! serve pra limpar a "lousa" com os erros do modelo antes de passar pra próxima imagem; assim, não mistura os erros da imagem anterior com os da seguinte
            saidas = model(images)
            perda = criterio(saidas, labels)
            perda.backward()    # o modelo vê o erro que cometeu e volta pra trás para encontrar onde ele errou
            otimizador.step()   # após descobrir o erro, o otimizador resolve pra não rolar de novo
            perda_corrente += perda.item()
            
            j += 1      
            
            if j % 10 == 0:
                print(f"Remessa: {j}/{len(treino_loader)*(epoca+1)} - Perda: {perda.item(): .4f}")

        perda_media_treino = perda_corrente / len(treino_loader)
        perdas_treino.append(perda_media_treino)

        # agora, para a validação
        model.eval()
        soma_perda_validacao = 0.0
        with torch.no_grad():   # isso significa que não precisa gastar memória calculando o erro para aprender
            for images, labels in validacao_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()
                saidas = model(images)
                perda = criterio(saidas, labels)
                soma_perda_validacao += perda.item()
                
        perda_media_validacao = soma_perda_validacao / len(validacao_loader)
        redutor.step(perda_media_validacao) # redutor vai diminuir taxa de aprendizado caso pare de aumentar
        perdas_validacao.append(perda_media_validacao)
        print(f"Época: {epoca} - Perda: {perda_media_treino: .4f} | Perda Validação: {perda_media_validacao: .4f}")
        
        # agora salvamos os melhores parâmetros e definimos paradas mais cedo
        if perda_media_validacao < melhor_perda_validacao:
            melhor_perda_validacao = perda_media_validacao
            contador = 0
            torch.save(model.state_dict(),f'melhor_modelo.pt')
            print(f"Modelo sendo salvo na época {epoca}")
        else:
            contador += 1
            
        if contador >= paciencia:
            print("Parando mais cedo que o normal!")
            break
      
    # vamos agora fazer um gráfico com as perdas      
    plt.plot(perdas_treino, label="Perdas durante Treinamento")
    plt.plot(perdas_validacao, label="Perdas durante Validação")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    
    plt.legend()
    plt.title("Perdas no Treino vs na Validação")
    plt.show()
    
    # vamos carregar o melhor modelo para vermos os detalhes dele
    model = modelopneumonia()
    state_dict = torch.load('./melhor_modelo.pt', weights_only=True)
    model.load_state_dict(state_dict)
    print(model.to(device))
    
    model.eval()
    correto, total = 0, 0
    predicoes = []
    probabilidade = []  # para AUC-ROC
    rotulos = []
    
    with torch.no_grad():
        for images, labels in teste_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicao = (probs > 0.5).float().flatten()
            
            probabilidade.extend(probs.cpu().numpy().flatten())   # para o AUC-ROC
            
            predicoes.extend(predicao.cpu().numpy())
            rotulos.extend(labels.cpu().numpy())
            
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
    
    print(f"Acurácia total: {accuracy_score(rotulos, predicoes)*100:.2f}%")
    print(classification_report(rotulos, predicoes, target_names=['Normal', 'Pneumonia']))
    auc = roc_auc_score(rotulos, probabilidade)
    print(f"AUC-ROC: {auc: .2f}")
    
    
    # vamos, agora, tentar mostrar para onde a IA olhou
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    model.eval()    # não sei se precisa colocar de novo, mas na dúvida
    
    camadas = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=camadas)
    camera_cinza = cam(input_tensor=images, targets=[ClassifierOutputTarget(0)])
    
    camera_cinza = camera_cinza[0, :]
    
    imagem_rgb = images[0].cpu().numpy().transpose(1, 2, 0)
    
    imagem_rgb = np.repeat(imagem_rgb, 3, axis=2)
    
    visualizacao = show_cam_on_image(imagem_rgb, camera_cinza, use_rgb=True)
    
    plt.imshow(visualizacao)
    plt.axis("off")
    plt.show()
    
    plt.imsave("grad-cam.png", visualizacao)