#include<fstream>
#include<iostream>
#include<iomanip>
#include <vector>
#include<unistd.h>
using namespace std;

int suspension_time_part(int **A,int i,int j){
    return A[i][2*j+1];
}

int suspension_time(int **A, int i, int M_i){
    int result = 0;
    for(int j=0;j<M_i-1;j++){
        result+= suspension_time_part(A,i,j);
    }
    return result;
}

int execution_time_part(int **A,int i,int j){
    return A[i][2*j];
}

int execution_time(int **A, int i, int M_i){
    int result = 0;
    for(int j=0;j<M_i;j++){
        //cout<<execution_time_part(A,i,j)<<" ";
        result += execution_time_part(A,i,j);
    }
    //cout<<"excetion_time"<<index<<"="<<result;
    return result;
}


int S(int **A, int *T, int *D, int i, int j,int M_i){//i代表第几个task
    if(j% M_i != M_i-1){
        return suspension_time_part(A,i,j % M_i);
    }
    else if(j<=M_i){
        return T[i]-D[i];
    }
    else{
        int total_time=0;
        for(int index=0;index<M_i*2-1;index++){
            total_time += A[i][index];
        }
        return T[i]- total_time;
    }
}

int find_l_max(int **A, int *T, int *D, int i, int h, int t,int M_i){
    int result=0;
    int j=h;
    int temp = execution_time_part(A,i,j % M_i) + S(A,T,D,i,j,M_i);
    if(temp>t){
        if(h==0){
            return 0;
        }
        else return -1;
    }
    result+= execution_time_part(A,i,j % M_i)+S(A,T,D,i,j,M_i);
    while(result<=t){
        j+=1;
        result =result+ execution_time_part(A,i,j % M_i)+S(A,T,D,i,j,M_i);
    }

    return j-1;
}



int workload_function_part(int **A, int *T, int *D, int i, int h, int t, int M_i){
    int result = 0;
    int l_max = find_l_max(A,T,D,i,h,t,M_i);
    for(int index=h; index<l_max+1; index++){
        result += execution_time_part(A,i, index % M_i);
    }
    int time_part_one = execution_time_part(A,i,(l_max+1)%M_i);
    int time_part_two = t;
    for(int index=h; index<l_max+1; index++){
        int temp = execution_time_part(A,i, index % M_i)+ S(A,T,D,i,index,M_i);
        time_part_two -= temp;
    }

    if(time_part_one>time_part_two) return result+time_part_two;
    else return result+time_part_one;
}




int workload_function(int **A,int *T, int *D, int i, int t, int M_i){
    //cout<<"in wl func"<<endl;
    int result=0;
    //int Mi=(task_size+1)/2;
    for(int index=0; index<M_i; index++){
        if (result< workload_function_part(A,T,D,i,index,t,M_i))
            result = workload_function_part(A,T,D,i,index,t,M_i);
    }
    return result;
}

int worst_case_response_time(int **A, int *T, int *D, int k,  int M_i){
    //cout<<"in worst case"<<endl;
    int result = execution_time(A,k,M_i) + suspension_time(A,k,M_i);
    //cout<<"result of index"<< k <<" task="<<result<<endl;
    if(k==0) return result;


    while(1){
        int temp=0;
        for(int index=0; index<k; index++){
            temp += workload_function(A,T,D,index,result,M_i);
        }

        //cout<<"Temp="<<temp<<" ";
        if(result >= execution_time(A,k,M_i)+ suspension_time(A,k,M_i)+temp) break;
        else {
            result++;
            if(result>10000) break;
            else if(result>T[k]) break;
        }
    }
    return result;
}

void randomly_create(int task_num,int task_size){
    sleep(1.0);
    srand((int)time(0));
    ofstream ofile;               //定义输出文件
    ofile.open("/Users/startingfromsjtu/Desktop/1119test/mytestfile.txt");     //作为输出文件打开
    //int task_num=10;
    int M[task_num];

    //M[0]=2; M[1]=2; M[2]=2; M[3]=2; M[4]=2;
    int M_i=(task_size+1)/2;
    int segment_length_max=100;
    ofile<<task_num<<endl;   //标题写入文件
    for(int index=0;index<task_num;index++){
        ofile<<M_i<<" ";
    }
    ofile<<endl;

    for(int index=0;index<task_num;index++){//initialize T
        ofile<<0<<" ";
    }
    ofile<<endl;

    for(int index=0;index<task_num;index++){//initialize D
        ofile<<0<<" ";
    }
    ofile<<endl;


    for(int i=0;i<task_num;i++) {
        for(int task_index=0;task_index<M_i*2-1;task_index++){
            int random=rand()%segment_length_max;
            while(random == 0){
                random=rand()%segment_length_max;
            }
            ofile << random << " ";
        }
        ofile <<endl;
    }
    //file.seekg(10,ios::beg);


    ofile.close();                //关闭文件

}

bool compute_one_task_set(int my_task_num, int task_size, int utilization_rate){

    randomly_create(my_task_num,task_size);
    //int task_size=5;
    vector<int> inputs;
    ifstream ifile;               //定义输入文件
    ifile.open("/Users/startingfromsjtu/Desktop/1119test/mytestfile.txt");     //作为输入文件打开
    int i;

    while(ifile>>i) {
        //cout << i << " ";     //屏幕显示
        inputs.push_back(i);
    }
    //cout<<endl;

    int task_num=inputs[0];

    int M[task_num];int T[task_num]; int D[task_num];
    int **tasks = new int* [task_num];//initializing
    for(int i=0; i< task_num; i++) tasks[i] = new int[task_size];

    inputs.erase(inputs.begin());


    for(int index=0;index<task_num;index++){//第二行是M
        M[index]=inputs[0];//cout<<"M["<<index<<"]="<<M[index]<<" ";
        inputs.erase(inputs.begin());
    }


    for(int index=0;index<task_num;index++){//第三行是T
        T[index]=inputs[0];
        inputs.erase(inputs.begin());
    }

    for(int index=0;index<task_num;index++){//第四行是D
        D[index]=inputs[0];
        inputs.erase(inputs.begin());
    }


    int inputs_index=0;
    for(int task_index=0;task_index<task_num;task_index++){
        for(int j=0;j<M[task_index]*2-1;j++){
            tasks[task_index][j]=inputs[inputs_index++];
            cout<<"tasks[" << task_index <<"]["<<j<<"]="<<tasks[task_index][j]<<" ";
        }
    }

    ifile.close();                 //关闭文件

    for(int index=0;index<task_num;index++){//用 C_sum 计算 T D
        int C_sum = execution_time(tasks,index,M[index]);
        //cout<<"for index="<<index<<", C_sum="<<C_sum<<endl;
        T[index]= C_sum *100/utilization_rate;
        D[index]= C_sum *100/utilization_rate;
    }

    cout<<endl;
    for(int index=0;index<task_num;index++){
        cout<<"T["<<index<<"]="<<T[index]<<" ";
    }
    cout<<endl;

    bool pass= true;

    for(int index=0;index<task_num;index++){
        int result = worst_case_response_time(tasks, T, D, index, M[index]);
        cout<<"task "<<index<<" final_result="<<result<<endl;
        if(result>D[index]) {
            pass = false;
            //break;
        }
    }

    for(int i = 0; i < task_size; i++)
    {
        delete[] tasks[i];
    }
    delete[] tasks;

    return pass;
}

int compute_several_task_set(int task_size, int task_num, int utilization_rate, int set_num){
    int successful_set_num=0;
    for(int set_index=0;set_index<set_num;set_index++){
        bool set_result=compute_one_task_set(task_num,task_size,utilization_rate);
        if(set_result== true) successful_set_num++;
    }
    //cout<<"++="<<successful_set_num;
    double successful_rate = (double)successful_set_num/set_num;
    cout<<"successful_set_num="<<successful_set_num<<endl;
    cout<<"successful_rate="<<successful_rate<<endl;
    return successful_set_num;
}


int main() {
//    int task_size=3;
//    int task_num=5;
//    int utilization_rate=10;
//    int set_num=100;
    int successful_set_num[100];
    bool flag[100];
    for(int index=0;index<100;index++){
        flag[index]=false;
    }

    for(int utilization_rate=10;utilization_rate<11;utilization_rate=utilization_rate+2){
        successful_set_num[utilization_rate] = compute_several_task_set(3,5,utilization_rate,10000);
        flag[utilization_rate]=true;
    }


    ofstream ofile;               //定义输出文件
    ofile.open("/Users/startingfromsjtu/Desktop/1119test/mytestresult.txt");
    for(int index=0;index<100;index++) {
        if (flag[index] == true) {
            cout << "successful_set_num[" << index << "]=" << successful_set_num[index] << endl;
            ofile<<"""successful_set_num[" << index << "]=" << successful_set_num[index] << endl;
        }
    }

    ofile.close();
}
