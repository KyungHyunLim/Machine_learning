//MNIST
#include <stdio.h>
#include <time.h>
#include <ctype.h>
#include <iostream>

using namespace std;

#define NLayer 2			//layer 수
#define MLayerSize 1000		//layer가 가질수 있는 최대 뉴론수 + 1
#define m0	270				//0층
#define m1	10				//1층 <현재 최종 출력층>
#define N 785				//input vector수 + 1 (28x28)
#define LEARNINGRATE 0.35
#define RAND_MAX 1000
#define N_tr_examples 60000
#define N_te_examples 10000

int M[NLayer] = { m0 , m1 };//모든 층의 뉴론의 수를 가진 배열
double s[NLayer][MLayerSize];
double f[NLayer][MLayerSize];
double delta[NLayer][MLayerSize];
double w[NLayer][MLayerSize][MLayerSize]; // 층수-뉴론번호-가중치번호

double input[N];	//input vector
double D[m1];		//정답레이블 최종층의 출력수 만큼
double TrainData[N_tr_examples][N - 1], d_tr[N_tr_examples][m1];	//훈련데이터 및 결과값
double TestData[N_te_examples][N - 1], d_te[N_te_examples][m1];	//테스트데이터 및 결과값

void datastream_tr();
void datastream_te();
void initializeW();					//weight 초기화
void initializeInput(int Trainnum);	//TrainData로부터 input vector값입력
void initializeD(int Trainnum);		//d_tr로부터 D vector값 입력
void initializeInput_te(int Testnum);	//TestData로부터 input vector값입력
void initializeD_te(int Testnum);		//d_te로부터 D vector값 입력
void forwardcomputation();
void backwardcomputation();
void weightupdate();
double avg_sq_error();
int epoch = 1;

void main() {
	
	FILE *out;
	int num_correct = 0;
	int pre_layer;
	int t, i, k;
	double temp = 0;
	int trainnum = 0;
	double result;
	int desire_ans = 0;
	int sys_ans = 0;
	double avgsq_error = 100.0;

	//파일 입출력
	datastream_tr();
	datastream_te();

	printf("\nSucces! data insert\n");
	//LEARNING
	initializeW();
	while (1) {
		for (trainnum = 0; trainnum < N_tr_examples; trainnum++) {
			//printf("epoch : %d, trainnum: %d\n", epoch, trainnum);
			//printf("preerrorrate : %lf", avgsq_error);
			initializeInput(trainnum);
			initializeD(trainnum);
			forwardcomputation();
			backwardcomputation();
			weightupdate();
		}
		avgsq_error = avg_sq_error();
		trainnum++;	
		epoch++;

		//TEST
		num_correct = 0;
		for (t = 0; t < N_te_examples; t++) {
			initializeInput_te(t);
			initializeD_te(t);
			forwardcomputation();
			temp = 0;
			for (int iz = 0; iz < m1; iz++) {	//확률(f가)이 가장 높은 값을 찾는다
				if (temp < f[1][iz]) {
					temp = f[1][iz];
					sys_ans = iz;
				}
			}
			for (int ix = 0; ix < m1; ix++) {	//답인 부분은 0이 아니므로. (원핫인코딩)
				if (d_te[t][ix] == 1) {
					desire_ans = ix;
					break;
				}
			}
			if (sys_ans == desire_ans)			//라벨과 동일하면 맞은 개수 증가
				num_correct++;
		}

		result = ((double)num_correct / (double)N_te_examples) * 100.0;
		cout << endl << "epoch : " << epoch - 1 << ", avgsq_error : " << avgsq_error << ", 정답률 : " << result;
		//printf_s("\nepoch : %d, 정답률 : %f\n",epoch-1, result);

		if (result > 99.5) {
			break;
		}
	}
	
	/* 결과저장.
	out = fopen("D:\\Programing\\c, c++\\MachineLearning\\result.txt" ,"w");
	fprintf(out, "layer#: %d\n" , NLayer);
	fprintf(out, "neuron# \nm0: %d, m1: %d\n", m0, m1);
	for (t = 0; t < NLayer; t++) {
		for (i = 0; i < M[t]; i++) {
			if (i == 0) {
				pre_layer = N;
			}
			else {
				pre_layer = M[i - 1] + 1;
			}
			for (k = 0; k < pre_layer; k++) {
				fputc(w[t][i][k], out);
				if(k%28 == 0)
					fputc('\n', out);
			}
		}
		fputc('\n', out);
	}

	printf("\n");
	*/
}

void initializeW() {
	int i, j, k, pre_layer, r;

	srand(time(NULL));

	for (i = 0; i < NLayer; i++) {
		//printf("\rinitialize_%dLayer", i);
		for (j = 0; j < M[i]; j++) {
			if (i == 0) {//0층 뉴론 (input vector로 부터 입력 받음)
				pre_layer = N;
			}
			else {//0층 이외의 뉴론 (아래층으로 부터 입력받음)
				pre_layer = M[i - 1] + 1;
			}
			for (k = 0; k < pre_layer; k++) {
				r = rand()% RAND_MAX;
				w[i][j][k] = ((double)r / (double)RAND_MAX ) - 0.5;
			}
		}
	}
	//printf("\n");
}
void forwardcomputation() {
	int i, j, L;
	//0층 계산
	for (i = 0; i < M[0]; i++) {
		s[0][i] = 0.0;
		for (j = 0; j < N; j++) {
			s[0][i] += input[j] * w[0][i][j];
		}
		f[0][i] = 1.0 / (1.0 + exp(-s[0][i]));
		f[0][m0] = 1.0;
	}
	//0층이후 계산
	for (L = 1; L < NLayer; L++) {		
		for (i = 0; i < M[L]; i++) {
			//printf("\rforwardcomputation_%dLayer - %dth neuron", L,i);
			s[L][i] = 0.0;
			for (j = 0; j < (M[L-1]+1); j++) {
				s[L][i] += f[L-1][j] * w[L][i][j];
			}
			//Sigmoid
			f[L][i] = 1.0 / (1.0 + exp(-s[L][i]));
		}
		f[L][M[L]] = 1.0;
	}
	//printf("\n");
}
void backwardcomputation() {
	int k = NLayer - 1; //최종층 번호
	int i, j, L;
	double tusum;
	//최종층 계산
	for (i = 0; i < M[k]; i++) {
		delta[k][i] = (D[i] - f[k][i])*(f[k][i])*(1 - f[k][i]);
	}
	//최종층 아래층 계산
	for (L = (NLayer - 2); L >= 0; L--) {	
		for (i = 0; i < M[L]; i++) {
			//printf("\rbackwardcomputation_%dLayer - %dth neuron", L,i);
			tusum = 0.0;
			for (j = 0; j < M[L + 1]; j++) {
				tusum += delta[L + 1][j] * w[L + 1][j][i];
			}
			delta[L][i] = f[L][i] * (1 - f[L][i])*tusum;
		}
	}
	//printf("\n");
}
void weightupdate() {
	int i, j, L;
	L = 0;
	double c = 0.05;	//learning rate
	//L=0
	for (i = 0; i < M[0]; i++) {
		for (j = 0; j < N; j++) {
			w[0][i][j] += LEARNINGRATE*delta[0][i] * input[j];
		}
	}

	//L>0
	for (L = 1; L < NLayer; L++) {		
		for (i = 0; i < M[L]; i++) {
			//printf("\rweightupdate_%dLayer - %dth neuron", L,i);
			for (j = 0; j < (M[L - 1] + 1); j++)
				w[L][i][j] += c*delta[L][i] * f[L - 1][j];
		}
	}

	//printf("\n");
}
double avg_sq_error() {
	double sum_sq_error = 0.0;
	int t, i;
	for (t = 0; t < N_tr_examples; t++) {
		initializeInput(t);
		initializeD(t);
		forwardcomputation();	
		//printf("now epoch : %d\n", epoch);
		//printf("%d / %d\n", t, N_tr_examples);
		//printf("sum_sq_error : %.2lf\n", sum_sq_error);
		for (i = 0; i < M[NLayer - 1]; i++) {
			sum_sq_error +=(double)((D[i] - f[NLayer - 1][i])*(D[i] - f[NLayer - 1][i]));
		}
	}
	sum_sq_error = sum_sq_error / (double)(N_tr_examples * M[NLayer - 1]);
	//printf("error : %f\n", sum_sq_error);
	return sum_sq_error;
}
void initializeInput(int Trainnum) {
	int i;
	for (i = 0; i < N-1; i++) {
		//printf("\rinitializeInput_%d", i);
		input[i] = TrainData[Trainnum][i];
	}
	input[N - 1] = 1;
	//printf("\n");
}
void initializeD(int Trainnum) {
	int i;
	for (i = 0; i < m1; i++) {
		//printf("\rinitializeD_%d", i);
		D[i] = d_tr[Trainnum][i];
	}
	//printf("\n");
}
void initializeInput_te(int Testnum) {
	int i;
	for (i = 0; i < N - 1; i++) {
		//printf("\rinitializeInput_te_%d", i);
		input[i] = TestData[Testnum][i];
	}
	input[N - 1] = 1;
	//printf("\n");
}
void initializeD_te(int Testnum) {
	int i;
	for (i = 0; i < m1; i++) {
		//printf("\rinitializeD_te_%d", i);
		D[i] = d_te[Testnum][i];
	}
	//printf("\n");
}
void datastream_tr() {
	FILE* fp1 = fopen("D:\\Programing\\c,c++\\MachineLearning\\train.txt", "r");
	char temp[150];
	char c = NULL;
	double digit;
	int z = 0;

	for (int i = 0; i < N_tr_examples; i++) {
		for (int p = 0; p < 29; p++) {
			fgets(temp, 100, fp1);
			if (strlen(temp) == 2) {//개행문자까지 2개..
				z = 0;
				char digit_[2];
				digit_[0] = temp[0];
				digit_[1] = '\0';
				digit = atof(digit_);
				for (int j = 0; j < 10; j++) {
					if (j == digit)
						d_tr[i][j] = 1.0;
					else
						d_tr[i][j] = 0.0;
				}
			}
			else {
				int h = 0;
				int g = 0;

				char cg_f[10];
				while (temp[h] != NULL && temp[h] != '\n') {
					if (temp[h] == ' ') {
						cg_f[g] = '\0';
						TrainData[i][z] = atof(cg_f) / 255.0;
						g = 0;
						z++;
						h++;
					}
					else {
						cg_f[g] = temp[h];
						g++;
						h++;
					}
				}
			}
		}
	}
}
void datastream_te() {
	FILE* fp2 = fopen("D:\\Programing\\c,c++\\MachineLearning\\test.txt", "r");
	char temp[150];
	double digit;
	int z = 0;;

	for (int i = 0; i < N_te_examples; i++) {
		for (int p = 0; p < 29; p++) {
			fgets(temp, 100, fp2);
			if (strlen(temp) == 2) {//개행문자까지 2개..
				z = 0;
				char digit_[2];
				digit_[0] = temp[0];
				digit_[1] = '\0';
				digit = atof(digit_);
				for (int j = 0; j < 10; j++) {
					if (j == digit)
						d_te[i][j] = 1.0;
					else
						d_te[i][j] = 0.0;
				}
			}
			else {
				int h = 0;
				int g = 0;

				char cg_f[10];
				while (temp[h] != NULL && temp[h] != '\n') {
					if (temp[h] == ' ') {
						cg_f[g] = '\0';
						TestData[i][z] = atof(cg_f) / 255.0;
						g = 0;
						z++;
						h++;
					}
					else {
						cg_f[g] = temp[h];
						g++;
						h++;
					}
				}
			}
		}
	}
}
