import cv2
import numpy as np

def aplicar_filtro_sepia(imagem):
    filtro_sepia = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])

    img = cv2.imread(imagem, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Imagem não encontrada!")

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        sepia = cv2.transform(bgr, filtro_sepia)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        img_sepia = cv2.merge((sepia, alpha))
    else:
        img_sepia = cv2.transform(img, filtro_sepia)
        img_sepia = np.clip(img_sepia, 0, 255).astype(np.uint8)

    return img_sepia

cv2.imwrite("arara_sepia.png", aplicar_filtro_sepia("arara.png"))

def aplicar_filtro_pb(imagem):
    img = cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem não encontrada!")
    return img

cv2.imwrite("arara_blackAndWhite.png", aplicar_filtro_pb("arara.png"))

def aplicar_filtro_negativo(imagem):
    img = cv2.imread(imagem, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Imagem não encontrada!")

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        negativo = 255 - bgr
        img_negativo = cv2.merge((negativo, alpha))
    else:
        img_negativo = 255 - img

    return img_negativo

cv2.imwrite("arara_invertido.png", aplicar_filtro_negativo("arara.png"))

def aplicar_filtro_blur(imagem, intensidade=5):
    img = cv2.imread(imagem, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Imagem não encontrada!")

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        blur = cv2.GaussianBlur(bgr, (intensidade, intensidade), 0)
        img_blur = cv2.merge((blur, alpha))
    else:
        img_blur = cv2.GaussianBlur(img, (intensidade, intensidade), 0)

    return img_blur

cv2.imwrite("arara_borrado.png", aplicar_filtro_blur("arara.png"))

def aplicar_filtro_oleo(imagem):
    img = cv2.imread(imagem)
    if img is None:
        raise ValueError("Imagem não encontrada!")

    img_oleo = cv2.xphoto.oilPainting(img, 7, 1)
    return img_oleo

cv2.imwrite("arara_oleo.png", aplicar_filtro_oleo("arara.png"))

def aplicar_filtro_lapis(imagem):
    img = cv2.imread(imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem não encontrada!")

    img_invertido = 255 - img
    img_blur = cv2.GaussianBlur(img_invertido, (21, 21), 0)
    img_sketch = cv2.divide(img, 255 - img_blur, scale=256)
    return img_sketch

cv2.imwrite("arara_lapis.png", aplicar_filtro_lapis("arara.png"))

def aplicar_filtro_cartoon(imagem):
    img = cv2.imread(imagem)
    if img is None:
        raise ValueError("Imagem não encontrada!")

    img_color = cv2.bilateralFilter(img, 9, 75, 75)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.adaptiveThreshold(img_gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
    img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)
    return img_cartoon

cv2.imwrite("arara_cartoon.png", aplicar_filtro_cartoon("arara.png"))
