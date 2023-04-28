
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<unistd.h>
using namespace std;


#define pi 3.1415926535897932384626433832795

bool L1_flag=1;

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	double res=0;
    for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}

string version;
char buf[100000],buf1[100000],buf2[100000];
int relation_num,entity_num;
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;

map<int,map<int,int> > left_entity,right_entity;
map<int,double> left_num,right_num;
//mamba
int type_num;
map<string,int> type2id;
map<int,string> id2type;
map<string, string> relation2headType,relation2tailType;

class Train{

public:
	map<pair<int,int>, map<int,int> > ok;
    void add(int x,int y,int z)
    {
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
        ok[make_pair(x,z)][y]=1;
    }
    void run(int n_in,double rate_in,double margin_in,int method_in)
    {
        n = n_in;
        rate = rate_in;
        margin = margin_in;
        method = method_in;
        relation_vec.resize(relation_num);
		for (int i=0; i<relation_vec.size(); i++)
			relation_vec[i].resize(n);
        entity_vec.resize(entity_num);
		for (int i=0; i<entity_vec.size(); i++)
			entity_vec[i].resize(n);
        relation_tmp.resize(relation_num);
		for (int i=0; i<relation_tmp.size(); i++)
			relation_tmp[i].resize(n);
        entity_tmp.resize(entity_num);
		for (int i=0; i<entity_tmp.size(); i++)
			entity_tmp[i].resize(n);
        /* 
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]);
        } */
        FILE* f1 = fopen(("relation2vec."+version).c_str(),"r");
        FILE* f3 = fopen(("entity2vec."+version).c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<endl;
        for (int i=0; i<relation_num;i++)
        {
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
            	cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        fclose(f1);
        fclose(f3);
        //mamba
        A.resize(type_num);
		for (int i=0; i<A.size(); i++)
			A[i].resize(n);
        A_tmp.resize(type_num);
		for (int i=0; i<A_tmp.size(); i++)
			A_tmp[i].resize(n);
        for (int i=0; i<type_num; i++)
        {
            for (int ii=0; ii<n; ii++){
				A[i][ii] = 1;
				A_tmp[i][ii] = 1;
			}
			//norm(A[i]);
        }


        A2.resize(relation_num);
		for (int i=0; i<A2.size(); i++)
			A2[i].resize(n);
        A2_tmp.resize(relation_num);
		for (int i=0; i<A2_tmp.size(); i++)
			A2_tmp[i].resize(n);
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++){
                A2[i][ii] = 1;
				A2_tmp[i][ii] = 1;
            }


        }

        bfgs();
    }

private:
    int n,method;
    double res;//loss function value
    double count,count1;//loss function gradient
    double rate,margin;
    double belta;
    vector<int> fb_h,fb_l,fb_r;
    vector<vector<int> > feature;
    vector<vector<double> > relation_vec,entity_vec;
    vector<vector<double> > relation_tmp,entity_tmp;
	vector<vector<double> > A,A_tmp;
	vector<vector<double> > A2,A2_tmp;
    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }

    void bfgs()
    {
        res=0;
        int nbatches=100;
        int nepoch = 1000;
        int batchsize = fb_h.size()/nbatches;
            for (int epoch=0; epoch<nepoch; epoch++)
            {

            	res=0;
             	for (int batch = 0; batch<nbatches; batch++)
             	{
             		relation_tmp=relation_vec;
            		entity_tmp = entity_vec;
					A_tmp = A;
					A2_tmp = A2;
             		for (int k=0; k<batchsize; k++)
             		{
						int i=rand_max(fb_h.size());
						string relation = id2relation[i];
						string headType = relation2headType[relation];
						string tailType = relation2tailType[relation];

						int j=rand_max(entity_num);
						double pr = 1000*right_num[fb_r[i]]/(right_num[fb_r[i]]+left_num[fb_r[i]]);
						if (method ==0)
                            pr = 500;
						if (rand()%1000<pr)
						{
							while (ok[make_pair(fb_h[i],fb_r[i])].count(j)>0)
								j=rand_max(entity_num);
							train_kb(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i]);
						}
						else
						{
							while (ok[make_pair(j,fb_r[i])].count(fb_l[i])>0)
								j=rand_max(entity_num);
							train_kb(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i]);
						}
                		norm(relation_tmp[fb_r[i]]);
                		norm(entity_tmp[fb_h[i]]);
                		norm(entity_tmp[fb_l[i]]);
                		norm(entity_tmp[j]);
						//norm(A_tmp[type2id[headType]]);
						//norm(A_tmp[type2id[tailType]]);

             		}
		            relation_vec = relation_tmp;
		            entity_vec = entity_tmp;
					A = A_tmp;
					A2 = A2_tmp;
             	}
                cout<<"epoch:"<<epoch<<' '<<res<<endl;
                FILE* f2 = fopen(("relation2vec."+version).c_str(),"w");
                FILE* f3 = fopen(("entity2vec."+version).c_str(),"w");
                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                    fprintf(f2,"\n");
                }
                for (int i=0; i<entity_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                    fprintf(f3,"\n");
                }
                fclose(f2);
                fclose(f3);
                //mamba
				FILE* f4 = fopen(("type2vec."+version).c_str(),"w");
                for (int i=0; i<type_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f4,"%.6lf\t",A[i][ii]);
                    fprintf(f4,"\n");
                }
				fclose(f4);

				FILE* f5 = fopen(("relationMatrix2vec."+version).c_str(),"w");
                for (int i=0; i<relation_num; i++)
                {
                    for (int ii=0; ii<n; ii++)
                        fprintf(f5,"%.6lf\t",A2[i][ii]);
                    fprintf(f5,"\n");
                }
				fclose(f4);
            }
    }
    double res1;
    double calc_sum2(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            	sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
        	for (int ii=0; ii<n; ii++)
            	sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        return sum;
    }
    void gradient2(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        for (int ii=0; ii<n; ii++)
        {

            double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel_a][ii]-=-1*rate*x;
            entity_tmp[e1_a][ii]-=-1*rate*x;
            entity_tmp[e2_a][ii]+=-1*rate*x;
            x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel_b][ii]-=rate*x;
            entity_tmp[e1_b][ii]-=rate*x;
            entity_tmp[e2_b][ii]+=rate*x;
        }
    }
    double calc_sum(int e1,int e2,int rel)
    {
        //mamba
		string relation = id2relation[rel];
		string headType = relation2headType[relation];
		string tailType = relation2tailType[relation];
		int headTypeId = type2id[headType];
		int tailTypeId = type2id[tailType];
		//cout << "rel" << rel << endl;
		//cout << "relation" << relation << endl;
		//cout << "headType" << headType << endl;
		//cout << "tailType" << tailType << endl;
		//cout << "headTypeId" << headTypeId << endl;
		//cout << "tailTypeId" << tailTypeId << endl;
		//cout << endl;
		//sleep(1);
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            	sum+=fabs(A[tailTypeId][ii]*entity_vec[e2][ii]-A[headTypeId][ii]*entity_vec[e1][ii]-A2[rel][ii]*relation_vec[rel][ii]);
        else
        	for (int ii=0; ii<n; ii++)
            	sum+=sqr(A[tailTypeId][ii]*entity_vec[e2][ii]-A[headTypeId][ii]*entity_vec[e1][ii]-A2[rel][ii]*relation_vec[rel][ii]);
        return sum;
    }
    void gradient(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
		string relation = id2relation[rel_a];
		string headType = relation2headType[relation];
		string tailType = relation2tailType[relation];
		int headTypeId = type2id[headType];
		int tailTypeId = type2id[tailType];

        for (int ii=0; ii<n; ii++)
        {

            double x = 2*(A[tailTypeId][ii]*entity_vec[e2_a][ii]-A[headTypeId][ii]*entity_vec[e1_a][ii]-A2[rel_a][ii]*relation_vec[rel_a][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel_a][ii]-=-1*rate*x*A2[rel_a][ii];
            entity_tmp[e1_a][ii]-=-1*rate*x*A[headTypeId][ii];
            entity_tmp[e2_a][ii]+=-1*rate*x*A[tailTypeId][ii];

			A_tmp[headTypeId][ii] -= -1*rate*x*entity_vec[e1_a][ii];
			A_tmp[tailTypeId][ii] += -1*rate*x*entity_vec[e2_a][ii];
			A2_tmp[rel_a][ii] -= -1*rate*x*relation_vec[rel_a][ii];



            x = 2*(A[tailTypeId][ii]*entity_vec[e2_b][ii]-A[headTypeId][ii]*entity_vec[e1_b][ii]-A2[rel_b][ii]*relation_vec[rel_b][ii]);
            if (L1_flag)
            	if (x>0)
            		x=1;
            	else
            		x=-1;
            relation_tmp[rel_b][ii]-=rate*x*A2[rel_b][ii];
            entity_tmp[e1_b][ii]-=rate*x*A[headTypeId][ii];
            entity_tmp[e2_b][ii]+=rate*x*A[tailTypeId][ii];

			A_tmp[headTypeId][ii] -= rate*x*entity_vec[e1_b][ii];
			A_tmp[tailTypeId][ii] += rate*x*entity_vec[e2_b][ii];
			A2_tmp[rel_b][ii] 	-= rate*x*relation_vec[rel_b][ii];

        }
    }
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        double sum1 = calc_sum(e1_a,e2_a,rel_a);
        double sum2 = calc_sum(e1_b,e2_b,rel_b);
        if (sum1+margin>sum2)
        {
        	res+=margin+sum1-sum2;
        	gradient( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
			
        }
    }
};

Train train;

void prepare()
{
	FILE* f0 = fopen("../data/relation_specific.txt","r");
	while (fscanf(f0,"%s%s%s",buf,buf1,buf2)==3)
	{
		string relation=buf;
		string headType=buf1;
		string tailType=buf2;
        relation2headType[relation] = headType;
        relation2tailType[relation] = tailType;
		//cout << "relation " << relation << " aaa" << endl;
		//cout << "headType " << headType << " aaa" << endl;
		//cout << "tailType " << tailType << " aaa" << endl;
		//cout << endl;
		//sleep(1);
	}

    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");

	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
    //mamba
	FILE* f3 = fopen("../data/type2id.txt","r");
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string st=buf;
		type2id[st]=x;
		id2type[x]=st;
		type_num++;
	}
    FILE* f_kb = fopen("../data/train.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        left_entity[relation2id[s3]][entity2id[s1]]++;
        right_entity[relation2id[s3]][entity2id[s2]]++;
        train.add(entity2id[s1],entity2id[s2],relation2id[s3]);
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	left_num[i]=sum2/sum1;
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	right_num[i]=sum2/sum1;
    }
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
    fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    int method = 1;
    int n = 200;
    double rate = 0.001;
    double margin = 1;
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
    cout<<"size = "<<n<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;
    if (method)
        version = "bern";
    else
        version = "unif";
    cout<<"method = "<<version<<endl;
    prepare();
    train.run(n,rate,margin,method);
    return 0;
}


