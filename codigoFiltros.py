import cv2
import numpy as np

def aplicar_filtro_sepia(imagem):
    # Matriz de transformação sépia
    filtro_sepia = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])

    # Lendo a imagem
    img = cv2.imread(imagem)
    
    # Aplicando o filtro
    img_sepia = cv2.transform(img, filtro_sepia)

    # Garantindo que os valores permaneçam dentro do intervalo válido (0 a 255)
    img_sepia = np.clip(img_sepia, 0, 255).astype(np.uint8)

    return img_sepia

# Salvar imagem
imagem_filtrada = aplicar_filtro_sepia("arara.jpg")
cv2.imwrite("arara_sepia.jpg", imagem_filtrada)

###########################################################################################

def aplicar_filtro_pb(imagem):
    # Lê a imagem já em escala de cinza
    img = cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)
    return img

# Salvar imagem
imagem_filtrada = aplicar_filtro_pb("arara.jpg")
cv2.imwrite("arara_blackAndWhite.jpg", imagem_filtrada)

###########################################################################################

def aplicar_filtro_negativo(imagem):
    # Lê a imagem em cores
    img = cv2.imread(imagem)

    # Inverte os valores dos pixels
    img_negativo = 255 - img
    return img_negativo

# Salvar imagem
imagem_filtrada = aplicar_filtro_negativo("arara.jpg")
cv2.imwrite("arara_invertido.jpg", imagem_filtrada)

###########################################################################################

def aplicar_filtro_blur(imagem, intensidade=5):
    # Lê a imagem
    img = cv2.imread(imagem)

    # Aplica desfoque gaussiano
    img_blur = cv2.GaussianBlur(img, (intensidade, intensidade), 0)
    return img_blur

# Salvar imagem
imagem_filtrada = aplicar_filtro_blur("arara.jpg")
cv2.imwrite("arara_borrado.jpg", imagem_filtrada)

###########################################################################################

def aplicar_filtro_oleo(imagem):
    # Lê a imagem
    img = cv2.imread(imagem)
    
    # Aplica o filtro de pintura a óleo na imagem, com parâmetros de raio (7) e intensidade (1)
    img_oleo = cv2.xphoto.oilPainting(img, 7, 1)
    return img_oleo

# Salvar imagem
imagem_filtrada = aplicar_filtro_oleo("arara.jpg")
cv2.imwrite("arara_oleo.jpg", imagem_filtrada)

###########################################################################################

def aplicar_filtro_lapis(imagem):
    # Lê a imagem em escala de cinza
    img = cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)

    # Inverte os tons da imagem
    img_invertido = 255 - img

    # Aplica desfoque gaussiano
    img_blur = cv2.GaussianBlur(img_invertido, (21, 21), 0)

    # Cria efeito de lápis dividindo os valores
    img_sketch = cv2.divide(img, 255 - img_blur, scale=256)
    return img_sketch

# Salvar imagem
imagem_filtrada = aplicar_filtro_lapis("arara.jpg")
cv2.imwrite("arara_lapis.jpg", imagem_filtrada)

###########################################################################################

def aplicar_filtro_cartoon(imagem):
    img = cv2.imread(imagem)
    img_color = cv2.bilateralFilter(img, 9, 75, 75)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
    img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)
    return img_cartoon

# salvar imagem
imagem_filtrada = aplicar_filtro_cartoon("arara.jpg")
cv2.imwrite("arara_cartoon.jpg", imagem_filtrada)
