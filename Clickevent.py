import cv2 as cv

clicks = []

def click_event(event, x, y, flags, params):
    """
    Click event where user clicks 4 points on the image. 
    """
    clicks, img = params["clicks"], params["img"]
    if (event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN) and len(clicks) < 4:
        print(x, y)
        clicks.append([x,y])
        print(clicks)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, ". " + f"{x},{y}", (x-6, y), font, 1, (255, 0, 0), 2)
        cv.imshow('image', img)
    
def get_four_clicks(image_path: str):
    """
    Shows an image and collects 4 left or right click points.

    Returns: a list of 4 points, where each point is a list of [x,y] coordinates.
    """
    clicks = []

    img = cv.imread(image_path, 1)
    if img is None:
        raise FileNotFoundError(f"Can't read the image: {image_path}")

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, "Click to Select 4 Corners", (0, 20), font, 0.5, (255, 0, 0), 2)

    cv.imshow("image", img)
    cv.setMouseCallback("image", click_event, {"clicks": clicks, "img": img})

    # Wait until we have 4 clicks 
    while len(clicks) < 4:
        cv.waitKey(20)
        
    cv.destroyAllWindows()
    print("Clicks:", clicks)

    return clicks

"""if __name__=="__main__":
    clicks = get_four_clicks('images/left01.jpg')"""