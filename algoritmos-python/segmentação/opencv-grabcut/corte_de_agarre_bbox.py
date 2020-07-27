# Como Usar:
# python corte_de_agarre_bbox.py

"""
N:.
│   corte_de_agarre_bbox.py
│   grabcut_mask.py
│
└───images
        adrian.jpg
        lighthouse.png
        lighthouse_mask.png
"""

# Importar os pacotes necessários
import numpy as np
import argparse
import time
import cv2
import os

# constrói o analisador de argumentos e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	default=os.path.sep.join(["images", "adrian.jpg"]),
	help="caminho para as imagens de entrada as quais nós vamos aplicar os cortes de agarre")
ap.add_argument("-c", "--iter", type=int, default=10,
	help="# interações do corte de agarre (grande valor => tempo de execução lento)")
args = vars(ap.parse_args())

# carregue a imagem de entrada do disco e aloque memória para o
# máscara de saída gerada pelo GrabCut - essa máscara deve ter o mesmo
# dimensões espaciais como imagem de entrada
image = cv2.imread(args["image"])
mask = np.zeros(image.shape[:2], dtype="uint8")

# define as coordenadas da caixa delimitadora que definem aproximadamente minha
# região do rosto e pescoço (ou seja, toda a pele visível)
rect = (151, 43, 236, 368)

# alocar memória para duas matrizes que o algoritmo GrabCut internamente
# usa ao segmentar o primeiro plano do segundo plano
fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")

# aplique GrabCut usando o método de segmentação de caixa delimitadora
start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
	fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_RECT)
end = time.time()
print("[INFO] aplicando corte de agarre tomando {:.2f} segundos".format(end - start))

# a máscara de saída possui possíveis valores de saída, marcando cada pixel
# na máscara como (1) plano de fundo definido, (2) primeiro plano definido,
# (3) plano de fundo provável e (4) plano de fundo provável
values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)

# loop sobre os possíveis valores da máscara GrabCut
for (name, value) in values:
	# constrói uma máscara que para o valor atual
	print("[INFO] mostrando máscara para '{}'".format(name))
	valueMask = (mask == value).astype("uint8") * 255

	# exibir a máscara para que possamos visualizá-la
	cv2.imshow(name, valueMask)
	cv2.waitKey(0)

# definiremos todos os pixels de fundo definidos e prováveis de fundo
# a 0 enquanto os pixels de primeiro plano e prováveis de primeiro plano são
# definido como 1
outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
	0, 1)

# dimensione a máscara do intervalo [0, 1] a [0, 255]
outputMask = (outputMask * 255).astype("uint8")

# aplique um AND bit a bit à imagem usando nossa máscara gerada por
# GrabCut para gerar nossa imagem final de saída
output = cv2.bitwise_and(image, image, mask=outputMask)

# mostra a imagem de entrada seguida pela máscara e pela saída gerada pelo
# GrabCut e mascaramento bit a bit
cv2.imshow("Entrada", image)
cv2.imshow("Máscara do corte de agarre", outputMask)
cv2.imshow("Saída do corte de agarre", output)
cv2.waitKey(0)
