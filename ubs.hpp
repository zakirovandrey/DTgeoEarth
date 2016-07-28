#ifndef UNIFORM_BETA_SPLINES
#define UNIFORM_BETA_SPLINES
// ВАЖНО параметр I НЕ является номером ячеки в которой проводится интеполяция
// это верхний край шаблона!!!
// Эта функция должна использоваться исключительно в обертках!
template <int degree> inline float unif_bspline(const float * A, const double & w, const int & I){
    float * prev, * cur;
    float a[degree], b[degree+1];
    prev = a;
    cur = b;
    for(int i = 0; i <= degree; i++){ // degree + 1
        for(int j = i; j <= degree; j++){ // degree + 1
            if (i==0) cur[j] = A[max(0,I+j-degree)]; // Нижний край, очевидно I-degree
            else {
                double alpha = (w+(degree -j))/(degree+1-i);
                cur[j-i] = (1-alpha)*prev[j -i] + alpha*prev[j-i+1];
            }
        }
        float * tmp=prev;
        prev = cur;
        cur  = tmp;
    }
    return prev[0];
}


// функция для проверки границ интерполяци. как и в пердыдущем случае I это верхняя граница шаблона
// N размер массива
template <int degree> inline bool check_borders(int I, int N){
    if ( I < degree || I >= N ) return false;//borders of array
    // bullshit
    else return true;
}
// приведение индексов к используемым в B-сплайнах
// ВНИМАНИЕ ЭТА ФУНКЦИЯ МЕНЯЕТ ПАРАМЕТРЫ
template <int degree> inline void index_recalculation(double & w, int & I){
    if (degree%2 == 0){
        I += w>=0.5 ? degree/2+1 : degree/2;
        w += w>=0.5 ? -0.5 : 0.5;
    } else I += (degree +1)/2;
}

template <int degree> inline float interpolate_ubs(const float * A, const double & w, const int & I){
    int myI = I; //copy
    double myw = w; //copy
    index_recalculation<degree>(myw,myI);
    return unif_bspline<degree>(A, myw, myI);
}

template <int degree> inline float interpolate_ubs_deriv(const float * A, const double &w, const double & _step, const int &I){
    int myI = I; //copy
    double myw = w; //copy
    index_recalculation<degree>(myw,myI);
    float der[degree];
    //auto _degstep=1.;
    //for(auto i =0; i< degree-1; i++) _degstep*=_step;
    for(int i =0; i< degree; i++) der[i] = ( A[myI+i-degree+1] - A[myI+i-degree] )*_step;
    return unif_bspline<degree-1>(der, myw, degree -1);
}

#endif // UNIFORM_BETA_SPLINES
