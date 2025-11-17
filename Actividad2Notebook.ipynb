import cv2 as cv
import numpy as np

# ---------------------------------------------------------
# 1. GRABCUT PARA GENERAR GROUND TRUTH
# ---------------------------------------------------------
def grabcut_auto(img_path):
    img = cv.imread(img_path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")

    h, w = img.shape[:2] #Se usan para calcular un rectángulo para GrabCut.

    rect = (int(w * 0.00),int(h * 0.20),int(w * 0.90),int(h * 0.60)) #Define rectángulo dónde está el objeto principal

    mask = np.zeros((h, w), np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # Ejecutar GrabCut
    cv.grabCut(img, mask, rect, bgModel, fgModel, 5, cv.GC_INIT_WITH_RECT)

    # Mascara final (foreground = 255)
    mask_final = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD),255,0).astype(np.uint8)

    # Imagen segmentada
    masked = cv.bitwise_and(img, img, mask=mask_final)

    # Mostrar ventanas solicitadas
    cv.imshow("Original", img)
    cv.imshow("Mascara GrabCut", mask_final)
    cv.imshow("Gato Segmentado", masked)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return mask_final


# ---------------------------------------------------------
# 2. EVALUACIÓN: IOU
# ---------------------------------------------------------
def evaluar_mascaras(pred_path, gt_mask):

    pred = cv.imread(pred_path, 0)
    if pred is None:
        raise ValueError("No se pudo cargar gato.png")

    if pred.shape != gt_mask.shape:
        raise ValueError("Las imágenes no tienen el mismo tamaño")

    # Convierte ambas a máscaras binarias
    _, pred_bin = cv.threshold(pred, 127, 1, cv.THRESH_BINARY)
    _, gt_bin = cv.threshold(gt_mask, 127, 1, cv.THRESH_BINARY)

    # intersección = píxeles donde ambas máscaras dicen “objeto”
    # unión = píxeles donde al menos una dice “objeto”
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    iou = intersection / union if union > 0 else 0
    dice = (2 * intersection) / (pred_bin.sum() + gt_bin.sum())

    return iou, dice


# ---------------------------------------------------------
# 3. PROGRAMA PRINCIPAL
# ---------------------------------------------------------
if __name__ == "__main__":

    # 1. Obtener máscara GrabCut (y mostrar imshow)
    gt_mask = grabcut_auto("gato.jpg")

    cv.imwrite("ground_truth_gc.png", gt_mask)
    print("Ground truth guardado como ground_truth_gc.png")

    # 2. Evaluar predicción
    iou, dice = evaluar_mascaras("gato.jpg", gt_mask)

    print("Resultados:")
    print("IoU:", iou)
    print("Dice:", dice)
