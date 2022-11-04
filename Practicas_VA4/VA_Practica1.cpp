////////////////////////////////Cabeceras/////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#define pi 3.1416
#define e 2.72
/////////////////////////////////////////////////////////////////////////////

///////////////////////////////Espacio de nombres////////////////////////////
using namespace cv;
using namespace std;
/////////////////////////////////////////////////////////////////////////////

//Funcion que imprime el tamaño de la imagen que se le envie
void SizeImg(Mat img) {
	cout << "Columnas: " << img.cols << endl;
	cout << "Filas: " << img.rows << endl;
	cout << "Tamaño: " << img.cols << "x" << img.rows << endl;
}

//Relleno de una imagen con bordes 0 para poder aplicar filtros sin perdida de calidad
Mat RefillImg(Mat original, int ksize) {
	//diferencia de pixeles en cada borde 
	int dif = (ksize - 1);
	Mat FilledImg(original.rows + dif, original.cols + dif, CV_8UC1);
	for (int i = 0; i < original.rows + dif;i++) {
		for (int j = 0; j < original.cols + dif;j++) {
			//Condicion que identifica si el pixel evauado es parte la imagen original
			if (i > dif / 2 && j > dif / 2 && i <= (original.rows - (dif / 2)) && j <= (original.cols - (dif / 2))) {
				FilledImg.at<uchar>(Point(i, j)) = uchar(original.at<uchar>(Point(i - (dif / 2), j - (dif / 2))));
			}
			//Si no es parte de la imagen original, entonces es borde y se le asign el valor de 0
			else {
				FilledImg.at<uchar>(Point(i, j)) = uchar(0);
			}
		}
	}
	return FilledImg;
}

//Creación de mascara para filtro gausseano
vector<vector<float>> Mask(int kSize, float sigma) {
	int difb = (kSize - 1) / 2;
	float operation = 0;
	//Vector de dos dimensiones con los valores del kernel
	vector<vector<float>> filter(kSize, vector<float>(kSize, 0));
	for (int i = -difb; i <= difb; i++)
	{
		for (int j = -difb; j <= difb; j++)
		{
			//Aplicamos la formula para cada pixel de nuestro kernel
			operation = (1 / (2 * pi * pow(sigma, 2))) * pow(e, -((pow(i,2) + pow(j,2)) / (2 * pow(sigma, 2))));
			filter[i + difb][j + difb] = operation;
			//Imprime valores del kernel 
			cout << operation << " ";
		}
		cout << endl;
	}
	return filter;
}

//Creación de Gx, mascara para sobel
vector<vector<float>> SobelY(int kSize) {
	vector<vector<float>> filter(kSize, vector<float>(kSize, 0));
	filter[0][0] = -1.0;
	filter[1][0] = 0.0;
	filter[2][0] = 1.0;
	filter[0][1] = -2.0;
	filter[1][1] = 0.0;
	filter[2][1] = 2.0;
	filter[0][2] = -1.0;
	filter[1][2] = 0.0;
	filter[2][2] = 1.0;


	return filter;
}
//Creación de Gy, mascara para sobel
vector<vector<float>> SobelX(int kSize) {
	vector<vector<float>> filterY(kSize, vector<float>(kSize, 0));
	filterY[0][0] = -1.0;
	filterY[1][0] = -2.0;
	filterY[2][0] = -1.0;
	filterY[0][1] = 0.0;
	filterY[1][1] = 0.0;
	filterY[2][1] = 0.0;
	filterY[0][2] = 1.0;
	filterY[1][2] = 2.0;
	filterY[2][2] = 1.0;


	return filterY;
}

//Convolucion del pixel evaluado en el filtro gausseano
float Filter(Mat original, vector<vector<float>> kernel, int kSize, int x, int y) {
	int difb = (kSize - 1) / 2;
	float sumPix = 0;
	float sumKernel = 0;
	for (int i = -difb; i <= difb; i++)
	{
		for (int j = -difb; j <= difb; j++)
		{
			float kernelVal = kernel[i + difb][j + difb];
			float imgVal = 0;
			imgVal = original.at<uchar>(Point(x+i, y+j));
			sumPix += (kernelVal * imgVal);
			sumKernel += kernelVal;
		}
	}
	return sumPix/sumKernel ;
}

//Convolucion del íxel evaluado en el filtro Sobel
float FilterSobel(Mat original, vector<vector<float>> kernel, int kSize, int x, int y) {
	int difb = (kSize - 1) / 2;
	float sumPix = 0;
	for (int i = -difb; i <= difb; i++)
	{
		for (int j = -difb; j <= difb; j++)
		{
			float kernelVal = kernel[i + difb][j + difb];
			float imgVal = 0;
			imgVal = original.at<uchar>(Point(x+i, y+j));
			sumPix += (kernelVal * imgVal);
		}
	}
	return sumPix;
}

//Recorrido de toda a imagen para hacer convolucion con filtro gauseano, implementado la funcion Filter
Mat FilterImgGaus(Mat ampliada, vector<vector<float>> kernel, int kSize, Mat gris_pond) {
	int difb = (kSize - 1) / 2;
	Mat filteredImg(gris_pond.rows, gris_pond.cols, CV_8UC1);
	int m = 0;
	for (int i = difb + 1; i < gris_pond.rows; i++)
	{
		int n = 0;
		for (int j = difb + 1; j < gris_pond.cols; j++) {
			filteredImg.at<uchar>(Point(m, n)) = uchar(Filter(ampliada, kernel, kSize, i, j));
			n++;
		}
		m++;
	}
	return filteredImg;
}

//Recorrido de toda a imagen para hacer convolucion con filtro Sobel, implementado la funcion FilterSobel
Mat FilterImgSobelG(Mat ampliada, vector<vector<float>> kernel, int kSize, Mat original) {
	int difb = (kSize - 1) / 2;
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	int m = 0;
	for (int i = difb + 1; i < original.rows; i++)
	{
		int n = 0;
		for (int j = difb + 1; j < original.cols; j++) {
			//Obtenemos el valor absoluto del pixel
			filteredImg.at<uchar>(Point(m, n)) = abs(static_cast<int>(FilterSobel(ampliada, kernel, kSize, i, j)));
			n++;
		}
		m++;
	}
	return filteredImg;
}


//Aplicamos funcion para combinar Gx y Gy del Sobel
Mat AplySobel(Mat Gx, Mat Gy) {
	double valSobel;
	double valGx, valGy;

	Mat imgSobel(Gx.rows, Gy.cols, CV_8UC1);

	for (int i = 0; i < Gx.rows; i++) {
		for (int j = 0; j < Gy.cols; j++) {

			valGx = Gx.at<uchar>(Point(j, i));
			valGy = Gy.at<uchar>(Point(j, i));

			valSobel = sqrt(pow(valGx, 2) + pow(valGy, 2));

			imgSobel.at<uchar>(Point(j, i)) = uchar(valSobel);
		}
	}
	return imgSobel;
}

//Con ayuda de arc tan y la funcion ista en clase, obtenemos la dirección hacia donde apunta el borde
vector<vector<double>> CalcularDir(Mat Gx, Mat Gy) {
	double valDir,valGx,valGy;
	vector<vector<double>> Direccion(Gx.rows, vector<double>(Gy.cols, 0));

	for (int i = 0; i < Gx.rows; i++) {
		for (int j = 0; j < Gy.cols; j++) {

			valGx = Gx.at<uchar>(Point(j, i));
			valGy = Gy.at<uchar>(Point(j, i));

			valDir = (atan(valGy / valGx) * 180.0) / pi;

			Direccion[i][j] = uchar(valDir);

			if (Direccion[i][j] < 0) {
				Direccion[i][j] += 180;
			}
		}
	}
	return Direccion;
}

//Obtenemos Non-maximum supression 
Mat supression(Mat imgSobel, vector<vector<double>> dir) {

	Mat imgSup(imgSobel.rows, imgSobel.cols, CV_8UC1);

	for (int i = 0; i < imgSobel.rows; i++) {
		for (int j = 0; j < imgSobel.cols; j++) {
			int lado1 = 255;
			int lado2 = 255;

			//Obtenemos valores de los lados
			if ((0 <= dir[i][j] < 22.5) || (157.5 <= dir[i][j] <= 180)) {
				lado1 = imgSobel.at<uchar>(Point(i, j + 1));
				lado2 = imgSobel.at<uchar>(Point(i, j - 1));
			}

			//Obtenemos esquinas
			else if (22.5 <= dir[i][j] < 67.5) {
				lado1 = imgSobel.at<uchar>(Point(i + 1, j - 1));
				lado2 = imgSobel.at<uchar>(Point(i - 1, j + 1));
			}

			//Obtenemos valores arriba y abajo
			else if (67.5 <= dir[i][j] < 112.5) {
				lado1 = imgSobel.at<uchar>(Point(i + 1, j));
				lado2 = imgSobel.at<uchar>(Point(i - 1, j));
			}

			//Obtenemos las qesquinas contrarias a las obtenidas en la primera condicion
			else if (112.5 <= dir[i][j] < 157.5) {
				lado1 = imgSobel.at<uchar>(Point(i - 1, j - 1));
				lado2 = imgSobel.at<uchar>(Point(i + 1, j + 1));
			}

			//Se mantienen las intensidades mayores a los ladoas
			if ((imgSobel.at<uchar>(Point(i, j)) >= lado1) && imgSobel.at<uchar>(Point(i, j)) >= lado2) {
				imgSup.at<uchar>(Point(i, j)) = imgSobel.at<uchar>(Point(i, j));
			}
			else {
				imgSup.at<uchar>(Point(i, j)) = uchar(0);
			}
		}
	}

	return imgSup;
}

//Obtenemos pixel maximo de la imagen
int max(Mat imgSup) {

	int max = 0;

	for (int i = 0; i < imgSup.rows; i++) {
		for (int j = 0; j < imgSup.cols; j++) {
			if (imgSup.at<uchar>(Point(i, j)) > max) {
				max = imgSup.at<uchar>(Point(i, j));
			}
		}
	}

	return max;
}

//Evaluamos si hay vecino 8conectado
bool weakVal(Mat Hyst, int x, int y) {
	int rows = Hyst.rows;
	int cols = Hyst.cols;
	int difb = 1;
	float pix = Hyst.at<uchar>(Point(x, y));
	for (int i = -difb; i <= difb; i++)
	{
		for (int j = -difb; j <= difb; j++)
		{
			float hystVal = 0;
			if (!(x + i < 0 || x + i >= cols || y + j < 0 || y + j >= rows)) {
				hystVal = Hyst.at<uchar>(Point(x + i, y + j));
				// si hay un vecino 8 conectado 
				if (hystVal == 255) {
					return true;
				}
			}

		}
	}
	return false;
}

//Aplicamos hysteresis
Mat hysteresis2(Mat imgSupr, float alto, float bajo) {
	Mat Hyst(imgSupr.rows, imgSupr.cols, CV_8UC1);
	Mat Hyst2(imgSupr.rows, imgSupr.cols, CV_8UC1);

	//Valores superior e inferior
	alto = max(imgSupr) * alto;
	bajo = alto * bajo;
	
	int debil = bajo;
	int fuerte = 255;
	int irr = 0;

	// Asignamos valores fuertes, debiles e irrrelevantes
	for (int i = 0; i < imgSupr.rows; i++) {
		for (int j = 0; j < imgSupr.cols; j++) {

			if (imgSupr.at<uchar>(Point(i, j)) >= alto) {
				Hyst.at<uchar>(Point(i, j)) = fuerte;
			}
			else if (bajo < imgSupr.at<uchar>(Point(i, j)) < alto) {
				Hyst.at<uchar>(Point(i, j)) = debil;
			}
			else {
				Hyst.at<uchar>(Point(i, j)) = irr;
			}
		}
	}

	for (int i = 0; i < imgSupr.rows; i++) {
		for (int j = 0; j < imgSupr.cols; j++) {
			//Volvemos pixeles debiles
			if (Hyst.at<uchar>(Point(i, j)) == debil) {
				if (weakVal(Hyst, i, j)) {
					Hyst2.at<uchar>(Point(i, j)) = fuerte;
				}
				else {
					Hyst2.at<uchar>(Point(i, j)) = irr;
				}
			}
			else {
				Hyst2.at<uchar>(Point(i, j)) = Hyst.at<uchar>(Point(i, j));
			}

		}
	}

	return Hyst2;
}

/////////////////////////Inicio de la funcion principal///////////////////
int main()
{

	/********Declaracion de variables generales*********/
	char NombreImagen[] = "lena.jpg";//Cargamos imagen
	Mat imagen; // Matriz que contiene nuestra imagen sin importar el formato
	/************************/

	/*********Lectura de la imagen*********/
	imagen = imread(NombreImagen);

	if (!imagen.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	/************************/

	/************Procesos*********/
	int fila_original = imagen.rows;
	int columna_original = imagen.cols;//Lectur de cuantas columnas
	int ksize = 0;
	float sigma = 0.0;

	//Matriz para imagen en escala de grises
	Mat gris_pond(fila_original, columna_original, CV_8UC1);

	//Obtenemos la imagen en escala de grises con valores ´ponderados
	for (int i = 0; i < fila_original; i++)
	{
		for (int j = 0; j < columna_original; j++)
		{
			double azul = imagen.at<Vec3b>(Point(j, i)).val[0];  // B
			double verde = imagen.at<Vec3b>(Point(j, i)).val[1]; // G
			double rojo = imagen.at<Vec3b>(Point(j, i)).val[2];  // R

			gris_pond.at<uchar>(Point(j, i)) = uchar(0.299 * rojo + 0.587 * verde + 0.114 * azul);
		}
	}


	//Ingresamos tamaño mascara
	cout << "Tamaño de la mascara: " << endl;
	cin >> ksize;

	//Verificamos que el tamaño sea impar
	if (ksize % 2 == 0) {
		cout << "Tamaño de mascara invalidoo" << endl;
		exit(0);
	}

	//Ingresamos valor de sigma
	cout << "Valor sigma: " << endl;
	cin >> sigma;

	//Rellenamos matriz en escala de grises con bordes de 0 dependiendo el tamaño de kernel
	Mat ImgRefill = RefillImg(gris_pond, ksize);
	//Obtenemos los valores del kernel
	vector<vector<float>> kernel = Mask(ksize, sigma);
	//Mandamos nuestros valores a la función que aplicca la convolución
	Mat filtrada = FilterImgGaus(ImgRefill, kernel, ksize,gris_pond);

	//Imagen ecualizada
	Mat ImagenEcualizada;
	equalizeHist(filtrada, ImagenEcualizada);

	//Obtenemos las mascaras Gx y Gy para implementar sobel
	vector<vector<float>> sobelx = SobelX(3);
	vector<vector<float>> sobely = SobelY(3);

	//Rellenamos con bordes la imagen ecualizada
	Mat ImgRefillS = RefillImg(ImagenEcualizada,3);

	//Convolucionamos Gx y Gy con nuestra imagen ecualizada
	Mat filtradaSobelGx = FilterImgSobelG(ImgRefillS,sobelx,3,ImagenEcualizada);
	Mat filtradaSobelGy = FilterImgSobelG(ImgRefillS, sobely, 3,ImagenEcualizada);
	//A traves de nuestra función juntamos ambas imagenes Gx y Gy para obtener el sobel
	Mat imgSobel = AplySobel(filtradaSobelGx,filtradaSobelGy);

	//Obtenemos gradiante (dirección)
	vector<vector<double>> dir = CalcularDir(filtradaSobelGx,filtradaSobelGy);

	//Obtenemos Non-maximum suppression
	Mat imgSupr = supression(imgSobel, dir);

	//Obtenemos hysteresis
	Mat imgHyst = hysteresis2(imgSupr, 0.5, 0.5);

	//Imprimimos tamaño de imagenes
	cout << "Tamaño original: " << endl;
	SizeImg(imagen);
	cout << endl;
	cout << "Tamaño escala grises: " << endl;
	SizeImg(gris_pond);
	cout << endl;
	cout << "Tamaño suavizada: " << endl;
	SizeImg(filtrada);
	cout << endl;
	cout << "Tamaño ecualizada: " << endl;
	SizeImg(ImagenEcualizada);
	cout << endl;
	cout << "Tamaño Sobel: " << endl;
	SizeImg(imgSobel);
	cout << endl;
	cout << "Tamaño Hysteresis: " << endl;
	SizeImg(imgHyst);
	cout << endl;




	
	namedWindow("Original", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Original", imagen);

	namedWindow("Escala_grises", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Escala_grises", gris_pond);

	namedWindow("Suavizada", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Suavizada", filtrada);

	namedWindow("Ecualizada", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Ecualizada", ImagenEcualizada);

	namedWindow("Sobel", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Sobel", imgSobel);

	namedWindow("Non_maximum_sup", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Non_maximum_sup", imgSupr);

	namedWindow("Hysteresis", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Hysteresis", imgHyst);






	/************************/

	waitKey(0); //Función para esperar
	return 1;
}


////////////////////////////////////////////////////////////////////////