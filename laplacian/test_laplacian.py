import sys
import cv2 as cv


def main(argv):

    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"

    imageName = argv[0] if len(argv) > 0 else 'lena.jpg'
    path = 'C:/Users/mentp/vscode/vscode-AI-workspace/laplacian/image/ex_1.jpg'
    
    src = cv.imread(path, cv.IMREAD_COLOR)

    if src is None:
        print('Error opening image')
        print('Program Arguments: [image_name -- default lena.jpg]')
        return -1

    #잡음 제거
    src = cv.GaussianBlur(src, (3, 3), 0)

    #회색으로 변경
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #윈도우창 생성
    cv.namedWindow(window_name,)
    #함수 호출
    dst = cv.Laplacian(src_gray,ddepth ,ksize=kernel_size)
    #CV_8U이미지로 출력 변화
    abs_dst = cv.convertScaleAbs(dst)

    #display
    cv.imshow(window_name, abs_dst)
    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])


