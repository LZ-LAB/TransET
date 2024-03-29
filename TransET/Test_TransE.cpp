#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<unistd.h>
using namespace std;

bool debug=false;
bool L1_flag=1;

string version;
string trainortest = "test";

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

int relation_num,entity_num;
int n= 200;

//mamba
int type_num;
map<string,int> type2id;
map<int,string> id2type;
map<string, string> relation2headType,relation2tailType;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%10==9)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000],buf2[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}

class Test{
    vector<vector<double> > relation_vec,entity_vec;
//mamba
	vector<vector<double> > A,A2;

    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double calc_sum(int e1,int e2,int rel)
    {
//mamba
		string relation = id2relation[rel];
		string headType = relation2headType[relation];
		string tailType = relation2tailType[relation];
		int headTypeId = type2id[headType];
		int tailTypeId = type2id[tailType];
		/*
		cout << "rel" << rel << endl;
		cout << "relation" << relation << endl;
		cout << "headType" << headType << endl;
		cout << "tailType" << tailType << endl;
		cout << "headTypeId" << headTypeId << endl;
		cout << "tailTypeId" << tailTypeId << endl;
		cout << endl;
		sleep(1);
		*/
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++){
				sum+=-fabs(A[tailTypeId][ii]*entity_vec[e2][ii]-A[headTypeId][ii]*entity_vec[e1][ii]-A2[rel][ii]*relation_vec[rel][ii]);
				
			}
            
        else
        for (int ii=0; ii<n; ii++){
			sum+=-sqr(A[tailTypeId][ii]*entity_vec[e2][ii]-A[headTypeId][ii]*entity_vec[e1][ii]-A2[rel][ii]*relation_vec[rel][ii]);
			
		}
            
        return sum;
    }
    void run()
    {
        FILE* f1 = fopen(("relation2vec."+version).c_str(),"r");
        FILE* f3 = fopen(("entity2vec."+version).c_str(),"r");
//mamba
		FILE* f4 = fopen(("type2vec."+version).c_str(),"r");
		FILE* f5 = fopen(("relationMatrix2vec."+version).c_str(),"r");
        cout<<relation_num<<' '<<entity_num<<' '<<type_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb;i++)
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
//mamba
		A.resize(type_num);
		for(int i = 0; i < type_num; i++)
		{
			A[i].resize(n);
			for(int ii = 0; ii < n; ii++)
			{
				fscanf(f4,"%lf",&A[i][ii]);
			}
		}
		
		A2.resize(relation_num);
		for(int i = 0; i < relation_num; i++)
		{
			A2[i].resize(n);
			for(int ii = 0; ii < n; ii++)
			{
				fscanf(f5,"%lf",&A2[i][ii]);
			}
		}
        fclose(f1);
        fclose(f3);
		fclose(f4);
		fclose(f5);
		double lsum=0 ,lsum_filter= 0;
		double rsum = 0,rsum_filter=0;
		double lp_n=0,lp_n_filter;
		double rp_n=0,rp_n_filter;
		
		double lp_10=0,lp_10_filter=0;
		double lp_3=0,lp_3_filter=0;
		double lp_1=0,lp_1_filter=0;
		double rp_10=0,rp_10_filter=0;
		double rp_3=0,rp_3_filter=0;
		double rp_1=0,rp_1_filter=0;
		double mrr =0,mrr_filter = 0;
		
		
		map<int,double> lsum_r,lsum_filter_r;
		map<int,double> rsum_r,rsum_filter_r;
		map<int,double> lp_n_r,lp_n_filter_r;
		map<int,double> rp_n_r,rp_n_filter_r;
		map<int,int> rel_num;
		double mrr_sum=0;
        for (int testid = 0; testid<fb_l.size(); testid+=1)
		{
			if(testid%100==0)	cout  << "testid" << testid << endl;
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
			double tmp = calc_sum(h,l,rel);
			rel_num[rel]+=1;
			vector<pair<int,double> > a;
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(i,l,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			double ttt=0;
			int filter = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(a[i].first,rel)].count(l)>0)
					ttt++;
			    if (ok[make_pair(a[i].first,rel)].count(l)==0)
			    	filter+=1;
				if (a[i].first ==h)
				{
					//a.size()-i就是这个三元组的排名
					//filter+1就是这个三元组在去除已有三元组的排名
					//mrr
                    mrr += 1.0 / (a.size()-i);
                    mrr_filter += 1.0 / (filter+1);
					//mr
					lsum+=a.size()-i;
					lsum_filter+=filter+1;
					
					lsum_r[rel]+=a.size()-i;
					lsum_filter_r[rel]+=filter+1;
					//Hits@10
					if (a.size()-i<=10)
					{
						lp_10+=1;
					}
					if (filter<10)
					{
						lp_10_filter+=1;
					}
					//Hits@3
					if (a.size()-i<=3)
					{
						lp_3+=1;
					}
					if (filter<3)
					{
						lp_3_filter+=1;
					}
					//Hits@1
					if (a.size()-i<=1)
					{
						lp_1+=1;
					}
					if (filter<1)
					{
						lp_1_filter+=1;
					}
					break;
				}
			}
			a.clear();
			for (int i=0; i<entity_num; i++)
			{
				double sum = calc_sum(h,i,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,rel)].count(a[i].first)>0)
					ttt++;
			    if (ok[make_pair(h,rel)].count(a[i].first)==0)
			    	filter+=1;
				if (a[i].first==l)
				{
					//mrr
                    mrr += 1.0 / (a.size()-i);
                    mrr_filter += 1.0 / (filter+1);
					//mr
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					
					rsum_r[rel]+=a.size()-i;
					rsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=10)
					{
						rp_n+=1;
						rp_n_r[rel]+=1;
					}
					if (filter<10)
					{
						rp_n_filter+=1;
						rp_n_filter_r[rel]+=1;
					}
					//Hits@10
					if (a.size()-i<=10)
					{
						rp_10+=1;
					}
					if (filter<10)
					{
						rp_10_filter+=1;
					}
					//Hits@3
					if (a.size()-i<=3)
					{
						rp_3+=1;
					}
					if (filter<3)
					{
						rp_3_filter+=1;
					}
					//Hits@1
					if (a.size()-i<=1)
					{
						rp_1+=1;
					}
					if (filter<1)
					{
						rp_1_filter+=1;
					}
					break;
				}
			}
        }
		//cout<<"left:"<<lsum/fb_l.size()<<'\t'<<lp_n/fb_l.size()<<"\t"<<lsum_filter/fb_l.size()<<'\t'<<lp_n_filter/fb_l.size()<<endl;
		//cout<<"right:"<<rsum/fb_r.size()<<'\t'<<rp_n/fb_r.size()<<'\t'<<rsum_filter/fb_r.size()<<'\t'<<rp_n_filter/fb_r.size()<<endl;
		cout << "raw MR " << (lsum + rsum)/(2 * fb_r.size()) << endl;
		cout << "filter MR " << (lsum_filter + rsum_filter)/(2 * fb_r.size()) << endl;
		
		cout << "raw MRR: " << mrr / (2 * fb_r.size()) << endl; 
		cout << "filter MRR: " << mrr_filter / (2 * fb_r.size()) << endl; 
		
		cout << "lp_10 " << lp_10 << endl;
		cout << "rp_10 " << rp_10 << endl;
		cout << "lp_10/fb_r.size() " << lp_10/fb_r.size() << endl;
		cout << "rp_10/fb_r.size() " << rp_10/fb_r.size() << endl;
		cout << "raw hits@10: " << (lp_10 + rp_10) / (2 * fb_r.size()) << endl;

		cout << "lp_3 " << lp_3 << endl;
		cout << "rp_3 " << rp_3 << endl;
		cout << "lp_3/fb_r.size() " << lp_3/fb_r.size() << endl;
		cout << "rp_3/fb_r.size() " << rp_3/fb_r.size() << endl;
		cout << "raw hits@3: " << (lp_3 + rp_3) / (2 * fb_r.size()) << endl;		

		cout << "lp_1 " << lp_1 << endl;
		cout << "rp_1 " << rp_1 << endl;
		cout << "lp_1/fb_r.size() " << lp_1/fb_r.size() << endl;
		cout << "rp_1/fb_r.size() " << rp_1/fb_r.size() << endl;
		cout << "raw hits@1: " << (lp_1 + rp_1) / (2 * fb_r.size()) << endl;

	
		cout << "lp_10_filter " << lp_10_filter << endl;
		cout << "rp_10_filter " << rp_10_filter << endl;
		cout << "lp_10_filter/fb_r.size() " << lp_10_filter/fb_r.size() << endl;
		cout << "rp_10_filter/fb_r.size() " << rp_10_filter/fb_r.size() << endl;
		cout << "filter hits@10: " << (lp_10_filter + rp_10_filter) / (2 * fb_r.size()) << endl;

		cout << "lp_3_filter " << lp_3_filter << endl;
		cout << "rp_3_filter " << rp_3_filter << endl;
		cout << "lp_3_filter/fb_r.size() " << lp_3_filter/fb_r.size() << endl;
		cout << "rp_3_filter/fb_r.size() " << rp_3_filter/fb_r.size() << endl;
		cout << "filter hits@3: " << (lp_3_filter + rp_3_filter) / (2 * fb_r.size()) << endl;

		cout << "lp_1_filter " << lp_1_filter << endl;
		cout << "rp_1_filter " << rp_1_filter << endl;
		cout << "lp_1_filter/fb_r.size() " << lp_1_filter/fb_r.size() << endl;
		cout << "rp_1_filter/fb_r.size() " << rp_1_filter/fb_r.size() << endl;
		cout << "filter hits@1: " << (lp_1_filter + rp_1_filter) / (2 * fb_r.size()) << endl;		

		
    }

};
Test test;

void prepare()
{
	FILE* f0 = fopen("../data/relation_specific.txt","r");
	while (fscanf(f0,"%s%s%s",buf,buf1,buf2)==3)
	{
		string relation=buf;
		string headType=buf1;
		string tailType=buf2;
        relation2headType[relation] = headType;
		/*
        relation2tailType[relation] = tailType;
		cout << "relation " << relation << endl;
		cout << "headType " << headType << endl;
		cout << "tailType " << tailType << endl;
		cout << endl;
		sleep(1);
		*/
		
	}
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		mid2type[st]="None";
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
	
    FILE* f_kb = fopen("../data/test.txt","r");
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
        	cout<<"miss relation:"<<s3<<endl;
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen("../data/train.txt","r");
	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
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

        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen("../data/valid.txt","r");
	while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
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
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb2);
}


int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        version = argv[1];
        prepare();
        test.run();
    }
}

