import cv2


def view(image: 'cv2.imread') -> None:
    cv2.imshow('image', image)
    cv2.waitKey(0)


def to_gray(image: 'cv2.imread') -> 'cv2.imread':
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_contours(gray_image: 'cv2.imread') -> 'cv2.imread':
    return cv2.Canny(gray_image, 50, 100)


def closing_image_apply(contours: 'cv2.imread') -> 'cv2.imread':
    return cv2.morphologyEx(contours, cv2.MORPH_CLOSE, (5, 5))


def find_contours(closing_image: 'cv2.imread') -> tuple:
    """Try to find external contours"""
    return cv2.findContours(closing_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


def objects_in_an_image(path: str) -> None:
    image: 'cv2.imread' = cv2.imread(path)
    resized: 'cv2.imread' = cv2.resize(image, (600, 500))
    gray = to_gray(resized)
    contours = get_contours(gray_image=gray)
    closing = closing_image_apply(contours)
    contours, hierachy = find_contours(closing)
    obj: int = 0
    for contourn in contours:
        cv2.drawContours(resized, contourn, -1, (0, 255, 0), 5)  # draw the contourn in objects
        # apply a rect in the image
        x, y, w, h = cv2.boundingRect(contourn)
        # save on file
        cv2.imwrite(f"objects/object{obj}.png", resized[y: y+h, x:w+x])
        # draw in the original image
        cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 3)
        print(x, y, w, h)
        obj += 1
    view(resized)


if __name__ == '__main__':
    objects_in_an_image('objetos.jpg')
