# Como Usar:
# python  mascara_corte_de_agarre.py

# importar os pacotes necessários
import numpy as np
import argparse
import time
import cv2
import os

# constrói o analisador de argumentos e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	default=os.path.sep.join(["images", "lighthouse.png"]),
	help="path to input image that we'll apply GrabCut to")
ap.add_argument("-mask", "--mask", type=str,
	default=os.path.sep.join(["images", "lighthouse_mask.png"]),
	help="path to input mask")
ap.add_argument("-c", "--iter", type=int, default=10,
	help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())

# carrega a imagem de entrada e a máscara associada do disco
image = cv2.imread(args["image"])
mask = cv2.imread(args["mask"], cv2.IMREAD_GRAYSCALE)

# aplique uma máscara bit a bit para mostrar qual seria a máscara aproximada aproximada
# que nos dá
roughOutput = cv2.bitwise_and(image, image, mask=mask)

# mostra a saída aproximada aproximada
cv2.imshow("Rough Output", roughOutput)
cv2.waitKey(0)

# qualquer valor de máscara maior que zero deve ser definido como provável
# primeiro plano
mask[mask > 0] = cv2.GC_PR_FGD
mask[mask == 0] = cv2.GC_BGD

# alocar memória para duas matrizes que o algoritmo Corte de Agarre internamente
# usa ao segmentar o primeiro plano do segundo plano
fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")

# aplique corte_de_agarre usando o método de segmentação por máscara
start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(image, mask, None, bgModel,
	fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_MASK)
end = time.time()
print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

# a máscara de saída possui possíveis valores de saída, marcando cada pixel
# na máscara como (1) plano de fundo definido, (2) primeiro plano definido,
# (3) plano de fundo provável e (4) plano de fundo provável
values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)

# loop sobre os possíveis valores da máscara Corte de Agarre
for (name, value) in values:
	# constrói uma máscara que para o valor atual
	print("[INFO] showing mask for '{}'".format(name))
	valueMask = (mask == value).astype("uint8") * 255

	# exibir a máscara para que possamos visualizá-la
	cv2.imshow(name, valueMask)
	cv2.waitKey(0)

# defina todos os pixels de fundo definidos e prováveis para 0
# enquanto os pixels de primeiro plano e prováveis de primeiro plano são definidos
# para 1, em seguida, dimensione a máscara do intervalo [0, 1] a [0, 255]
outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
	0, 1)
outputMask = (outputMask * 255).astype("uint8")

# aplique um AND bit a bit à imagem usando nossa máscara gerada por
# Corte de Agarre para gerar nossa imagem final de saída
output = cv2.bitwise_and(image, image, mask=outputMask)

# mostra a imagem de entrada seguida pela máscara e pela saída gerada pelo
# Corte de Agarre e mascaramento bit a bit
cv2.imshow("Input", image)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imshow("GrabCut Output", output)
cv2.waitKey(0)