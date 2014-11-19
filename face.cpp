#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;


//将给出的图像回归为值域在0~255之间的正常图像
Mat norm_0_255(const Mat& src) {
    // 构建返回图像矩阵
    Mat dst;
    switch(src.channels()) {
    case 1://根据图像通道情况选择不同的回归函数
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

// 将一副图像的数据转换为Row Matrix中的一行；这样做是为了跟opencv给出的PCA类的接口对应
//参数中最重要的就是第一个参数，表示的是训练图像样本集合
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // 样本个数
    size_t n = src.size();
    // 如果样本为空，返回空矩阵
    if(n == 0)
        return Mat();
    // 样本的维度
    size_t d = src[0].total();
    // 构建返回矩阵
    Mat data(n, d, rtype);
    // 将图像数据复制到结果矩阵中
    for(int i = 0; i < n; i++) {
        //如果数据为空，抛出异常
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // 图像数据的维度要是d，保证可以复制到返回矩阵中
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // 获得返回矩阵中的当前行矩阵:
        Mat xi = data.row(i);
       // 将一副图像映射到返回矩阵的一行中:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

int main(int argc, const char *argv[]) {
    // 训练图像集合
    vector<Mat> db;

    //使用ORL人脸库，可以自行在网上下载
    //将数据读入到集合中

    db.push_back(imread("s1.bmp", IMREAD_GRAYSCALE));
    db.push_back(imread("s2.bmp", IMREAD_GRAYSCALE));
    db.push_back(imread("s3.bmp", IMREAD_GRAYSCALE));

    db.push_back(imread("s4.bmp", IMREAD_GRAYSCALE));
    db.push_back(imread("s5.bmp", IMREAD_GRAYSCALE));
    db.push_back(imread("s6.bmp", IMREAD_GRAYSCALE));

    db.push_back(imread("s7.bmp", IMREAD_GRAYSCALE));
    db.push_back(imread("s8.bmp", IMREAD_GRAYSCALE));
    db.push_back(imread("s9.bmp", IMREAD_GRAYSCALE));

    db.push_back(imread("s10.bmp", IMREAD_GRAYSCALE));
    db.push_back(imread("s11.bmp", IMREAD_GRAYSCALE));
    db.push_back(imread("s12.bmp", IMREAD_GRAYSCALE));

    // 将训练数据读入到数据集合中，实现PCA类的接口
    Mat data = asRowMatrix(db, CV_32FC1);

    // PCA中设定的主成分的维度,这里我们设置为10维度
    int num_components = 10;

    // 构建一份PCA类
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    // 复制PCA方法获得的结果
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();

    // 平均脸:
    imshow("avg", norm_0_255(mean.reshape(1, db[0].rows)));

    // 前三个训练人物的特征脸
    imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, db[0].rows));
    imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, db[0].rows));

    // Show the images:
    waitKey(0);

    // Success!
    return 0;
}
