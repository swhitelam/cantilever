
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include <string.h>

using namespace std;

char st[256];
long seed;

//mmm, pi
const double pi=4.0*atan(1.0);

//production
const int q_production_run=0; //flag

//pre-trained?
const int q_read_start_net=1; //flag

//protocol type
const int q_protocol=2; //flag (0) basic, (1) enhanced, (2) net

//number of consecutive erasures
const int n_reps=1; //flag

//training
const int number_of_trajectories=10000; //flag
const int number_of_trajectories_histogram=number_of_trajectories;

//time parameters (fixed)
double timestep=0.1; //in microseconds //flag
int number_of_report_steps=1000;
const int max_number_of_report_steps=1000;

//time parameters (set in initialize)
double tf;
double tau_protocol;
double trajectory_time;
int trajectory_steps;
int erasure_type;
int report_step;

//lookup table
double** lookup_table;

//for averaging
double wd[number_of_trajectories*n_reps];
double he[number_of_trajectories];
double pos[number_of_trajectories];
double vel[number_of_trajectories];
double teff[number_of_trajectories]; //max value reached
double etot[number_of_trajectories];

double success[n_reps];

//potential parameters
double c0,c1;

//model parameters (kt, microseconds^-1, nanometers)
double kay=1.0;
double eff_0=1090; //in Hz
double quality=7.0; //flag
double omega_0=(2.0*pi*eff_0)/(1e6); //in reciprocal microseconds, assuming f_0=1200 Hz
double cycle_time=1e6/eff_0; //1/f_0, in microseconds

//boundary values
double c0_boundary=0.0;
double c1_boundary=5.0;

//vars
double position;
double velocity;

//plot potential
const int potential_pics=10;
int traj_number=0;
int potential_pic_counter=0;
double pos_time[number_of_trajectories][potential_pics+2];
double vel_time[number_of_trajectories][potential_pics+2];
double te_time[number_of_trajectories][potential_pics+2];

//general net parameters
const int depth=4; //stem of mushroom net, >=1 (so two-layer net at least)
const int width=4;
const int width_final=10;

//specific net parameters
const int number_of_nets=1;
const int number_inputs[number_of_nets]={1};
const int number_outputs[number_of_nets]={2};

//includes delta-function terms
const int net_params0=number_inputs[0]*width+width*width*(depth-1)+width*width_final+depth*width+width_final+(width_final+1)*number_outputs[0];
const int number_of_net_params[number_of_nets]={net_params0};
const int number_of_net_parameters=number_of_net_params[0];

//set within net function:
const int max_inputs=2;
const int max_outputs=2;

double inputs[max_inputs];
double outputs[max_outputs];

//GA hyperparms
double sigma_mutate=0.02;

double np;
double phi;

//registers
double* mutation = (double*) malloc(number_of_net_parameters * sizeof(double));
double* mean_mutation = (double*) malloc(number_of_net_parameters * sizeof(double));
double* net_parameters = (double*) malloc(number_of_net_parameters * sizeof(double));
double* net_parameters_holder = (double*) malloc(number_of_net_parameters * sizeof(double));

int initial_state;
int number_state_one;
double teff_time[max_number_of_report_steps*n_reps][2];
double work_time[max_number_of_report_steps*n_reps][2]; //starting states P,M
double energy_time[max_number_of_report_steps*n_reps][2];
double position_time[max_number_of_report_steps*n_reps][2];
double velocity_time[max_number_of_report_steps*n_reps][2];

double mean_total_energy;
double total_energy[max_number_of_report_steps*n_reps];
double kinetic_energy[max_number_of_report_steps*n_reps];
double potential_energy[max_number_of_report_steps*n_reps];

//vars int
int generation=0;
int record_trajectory=0;

double tau;
double heat;
double work;
double energy;
double min_var;
double mean_cs;
double mean_heat;
double mean_teff;
double mean_work;
double histo_width;
double mean_prob_erasure;
double mean_delta_position;


//functions void
void ga(void);
void read_net(void);
void averaging(void);
void store_net(void);
void initialize(void);
void output_net(void);
void mutate_net(void);
void restore_net(void);
void equilibrate(void);
void net_protocol(void);
void jobcomplete(int i);
void shift_c1(double dv);
void run_net(int net_id);
void initialize_net(void);
void langevin_step(int s1);
void reset_registers(void);
void final_potential(void);
void output_histogram(void);
void default_protocol(void);
void run_trajectory(int n1);
void set_erasure_type(int i);
void initial_potential(void);
void make_lookup_table(void);
void shift_velocity(double dv);
void run_trajectory_average(void);
void read_semi_optimal_protocol(void);
void record_position(int step_number);
void update_potential(int step_number);
void output_potential(int step_number);
void output_generational_data(int gen1);
void output_trajectory_average_data(void);
void output_histogram_velocity_norm(void);
void output_trajectory_data(int step_number);
void output_histogram_position(int time_slice);
void output_histogram_velocity(int time_slice);
void output_rep_data(int step_number, int rep_number);
void record_trajectory_averages(int step_number, int rep_number);

void plot_function(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max);
void plot_individual_function(const char *varname, const char *xlabel, int count1, double x_min, double x_max, const char *ylabel, double y_min, double y_max);

//functions int
int sign(double x);

//functions double
double test_phi(void);
double potential(void);
double heaviside(double x);
double gauss_rv(double sigma);

int main(void){
  
//RN generator
initialize();

//GA
ga();

return 0;

}

void initialize(void){

//clean up
snprintf(st,sizeof(st),"rm report_*");
cout << st << endl;
cout << endl;
system(st);

//set trajectory time
double en=1.0; //flag
trajectory_time=en*cycle_time+1.0*cycle_time; //wait for one period at end to check stability
trajectory_steps=(int) (trajectory_time/timestep);

//set protocol time
tau_protocol=(trajectory_time-1.0*cycle_time)/(trajectory_time);
tf=trajectory_time*tau_protocol;

//calculate report step
report_step= trajectory_steps / max_number_of_report_steps;
if(trajectory_steps % max_number_of_report_steps >0){report_step++;}
number_of_report_steps=(int) (trajectory_steps/report_step);
if(number_of_report_steps>max_number_of_report_steps){cout << " here 1 " << endl; exit(2);}

cout << " traj time " << trajectory_time << endl;
cout << " report step " << report_step << endl;
cout << " number of report steps " << number_of_report_steps << endl;
cout << " report time " << report_step*timestep*number_of_report_steps << endl;

//allocate memory for  lookup_table
lookup_table = new double*[trajectory_steps];
for(int i = 0;i<trajectory_steps;i++){lookup_table[i] = new double[2];}

ifstream infile0("input_parameters.dat", ios::in);
while (!infile0.eof ()){infile0 >> seed >> generation;}

if(generation>1){
snprintf(st,sizeof(st),"rm net_*_gen_%d.dat",generation-2);
cout << st << endl;
cout << endl;
system(st);
}

//seed RN generator
srand48(seed);

//initialize net
if(generation==0){

if(q_read_start_net==0){initialize_net();}
else{

snprintf(st,sizeof(st),"cp start_net.dat net_in_gen_0.dat");
system(st);

read_net();

}}
else{read_net();}


snprintf(st,sizeof(st),"zero.dat");
ofstream output_zero(st,ios::out);
output_zero << 0 << " " << 0 << endl;
output_zero << 0 << " " << 1 << endl;
output_zero.close();

}



double gauss_rv(double sigma){

double r1,r2;
double g1;
double two_pi = 2.0*pi;

r1=drand48();
r2=drand48();

g1=sqrt(-2.0*log(r1))*sigma*cos(two_pi*r2);

return (g1);

}


void update_potential(int step_number){

double e1,e2;
double s1=1.0-2.0*erasure_type;

//initial energy
e1=potential();

//protocol
if(step_number<trajectory_steps){c0=s1*lookup_table[step_number][0];c1=lookup_table[step_number][1];}
else{final_potential();}

//final energy
e2=potential();

//work
work+=e2-e1;

//log energy
energy=e2;

}



void run_trajectory(int n1){

int i;

//set erasure type
set_erasure_type(n1);

//initial position
if(n1==0){equilibrate();}

//run traj
for(i=0;i<=trajectory_steps;i++){

//output data
output_trajectory_data(i);

//output rep data
output_rep_data(i,n1);

//record trajectory averages
record_trajectory_averages(i,n1);

//update potential
if((i!=trajectory_steps) && (n1==0)){output_potential(i);}
update_potential(i);

//update position
langevin_step(i);

//record position
if(n1==0){record_position(i);}

//update time
if(i!=trajectory_steps){tau+=1.0/(1.0*trajectory_steps);}

}

//final-time data
if(n1==0){output_potential(trajectory_steps);}
output_trajectory_data(trajectory_steps);

//increment trajectory counter
if(n1==0){traj_number++;}


}

void langevin_step(int s1){

double e1,e2;

//initial energy
e1=potential();

//parameters
double lambda=exp(-timestep*omega_0/quality); //with timestep in microseconds

//gradient term
double grad=kay*(position-sign(position-c0)*c1);
//double grad=kay*(position-c0);
grad=grad*quality*omega_0; //Q omega_0= k/gamma

//position update
position=position+velocity*timestep;

//velocity update (nm/ms)
velocity=lambda*velocity-(1.0-lambda)*grad+sqrt(1.0-lambda*lambda)*omega_0*gauss_rv(1.0);

//final energy
e2=potential();

//heat increment
heat+=e2-e1;

}


void output_trajectory_data(int step_number){

int q_ok=0;

if(record_trajectory==1){

if(step_number % report_step==0){q_ok=1;}
if(step_number==0){q_ok=1;}
if(step_number==1){q_ok=1;}
if(step_number==trajectory_steps-2){q_ok=1;}
if(step_number==trajectory_steps-1){q_ok=1;}
if(step_number==trajectory_steps){q_ok=1;}

if(q_ok==1){

snprintf(st,sizeof(st),"report_position_gen_%d.dat",generation);
ofstream out1(st,ios::app);

snprintf(st,sizeof(st),"report_velocity_gen_%d.dat",generation);
ofstream out2(st,ios::app);

snprintf(st,sizeof(st),"report_work_gen_%d.dat",generation);
ofstream out3(st,ios::app);

snprintf(st,sizeof(st),"report_heat_gen_%d.dat",generation);
ofstream out4(st,ios::app);

snprintf(st,sizeof(st),"report_c0_gen_%d.dat",generation);
ofstream out5(st,ios::app);

snprintf(st,sizeof(st),"report_c1_gen_%d.dat",generation);
ofstream out6(st,ios::app);

snprintf(st,sizeof(st),"report_energy_gen_%d.dat",generation);
ofstream out7(st,ios::app);

snprintf(st,sizeof(st),"report_teff_gen_%d.dat",generation);
ofstream out8(st,ios::app);

out1 << tau << " " << position << endl;
out2 << tau << " " << velocity << endl;
out3 << tau << " " << work << endl;
out4 << tau << " " << heat << endl;
out5 << tau << " " << c0 << endl;
out6 << tau << " " << c1 << endl;
out7 << tau << " " << potential() << endl;
out8 << tau << " " << velocity*velocity/(omega_0*omega_0) << endl;

}}

}

void output_rep_data(int step_number, int rep_number){

int q_ok=0;

if(record_trajectory==1){

if(step_number % report_step==0){q_ok=1;}
if(step_number==0){q_ok=1;}
if(step_number==1){q_ok=1;}
if(step_number==trajectory_steps-2){q_ok=1;}
if(step_number==trajectory_steps-1){q_ok=1;}
if(step_number==trajectory_steps){q_ok=1;}

if(q_ok==1){

snprintf(st,sizeof(st),"report_rep_position_gen_%d.dat",generation);
ofstream out1(st,ios::app);

snprintf(st,sizeof(st),"report_rep_target_gen_%d.dat",generation);
ofstream out2(st,ios::app);

out1 << tau << " " << position << endl;
out2 << tau << " " << (2.0*erasure_type-1.0)*c1_boundary << endl;

}}

}


void plot_function(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max){

//snprintf(st,sizeof(st),"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/config.asy ."); system(st);
//snprintf(st,sizeof(st),"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/graph_routines.asy ."); system(st);

const char *varname1="cee";
const char *varname1a="c0";
const char *varname1b="c1";

const char *varname3="position";
//const char *varname3b="energy";
//const char *varname3c="work";
//const char *varname3d="teff";

const char *varname5="wd";

const char *varname7="rep_position";
const char *varname7b="rep_target";

const char *varname6="energy";
const char *varname6a="total_energy";
const char *varname6b="kinetic_energy";
const char *varname6c="potential_energy";

const char *varname_work="work_by_well";
const char *varname_work_0="work_average_state_0";
const char *varname_work_1="work_average_state_1";

const char *varname_teff="teff_by_well";
const char *varname_teff_0="teff_average_state_0";
const char *varname_teff_1="teff_average_state_1";

const char *varname_pot="potential_by_well";
const char *varname_pot_0="energy_average_state_0";
const char *varname_pot_1="energy_average_state_1";


 //output file
 snprintf(st,sizeof(st),"report_%s.asy",varname);
 ofstream output_interface_asy(st,ios::out);
 
 //write output file
 output_interface_asy << "import graph;" << endl;
 output_interface_asy << "import stats;"<< endl;
 
 output_interface_asy << "from \"graph_routines.asy\" access *;"<< endl;
 
 output_interface_asy << "picture p2;"<< endl;
 output_interface_asy << "defaultpen(1.5);"<< endl;
 
 output_interface_asy << "real ymin=" << y_min << ";"<< endl;
 output_interface_asy << "real ymax=" << y_max << ";"<< endl;
 
 output_interface_asy << "real xmin=" << x_min << ";"<< endl;
 output_interface_asy << "real xmax=" << x_max << ";"<< endl;
 
 output_interface_asy << "size(p2,400,400,IgnoreAspect);"<< endl;
 output_interface_asy << "scale(p2,Linear(true),Linear(true));"<< endl;
 
 
 //void simplot_symbol(picture p, string filename,string name,pen pn,int poly,real a1,real a2,real a3,real s1)

if((varname!=varname1) && (varname!=varname3) &&(varname!=varname6) && (varname!=varname_work) && (varname!=varname_teff) && (varname!=varname_pot)){
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,generation,0.1,0.8,0.1,1.2);
output_interface_asy << st << endl;
}

if(varname==varname1){

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1a,generation,0.1,0.8,0.1,1.2);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1b,generation,0.1,0.1,0.8,0.9);
output_interface_asy << st << endl;

//snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"jumps.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",0.0,0.0,0.0,1.0);
//output_interface_asy << st << endl;

//snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1b,generation,0.8,0.1,0.1,1.2);
//output_interface_asy << st << endl;

}

if(varname==varname_work){

//work
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname_work_0,generation,0.2,0.2,0.8,1.2);
output_interface_asy << st << endl;
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname_work_1,generation,0.8,0.2,0.2,1.2);
output_interface_asy << st << endl;

}

if(varname==varname_teff){

//work
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname_teff_0,generation,0.2,0.2,0.8,1.2);
output_interface_asy << st << endl;
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname_teff_1,generation,0.8,0.2,0.2,1.2);
output_interface_asy << st << endl;

}

if(varname==varname_pot){

//work
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname_pot_0,generation,0.2,0.2,0.8,1.2);
output_interface_asy << st << endl;
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname_pot_1,generation,0.8,0.2,0.2,1.2);
output_interface_asy << st << endl;

}

if(varname==varname5){
//snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"opt.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",0.0,0.0,0.0,1.0);
//output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"zero.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",0.0,0.0,0.0,1.0);
output_interface_asy << st << endl;
}

if(varname==varname6){
snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname6a,generation,0.1,0.8,0.1,1.2);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname6b,generation,0.1,0.1,0.8,1.0);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname6c,generation,0.7,0.1,0.1,1.0);
output_interface_asy << st << endl;
}

if(varname==varname7){
snprintf(st,sizeof(st),"simplot_simple_dashed(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname7b,generation,0.0,0.0,0.0,1.0);
output_interface_asy << st << endl;
}


 output_interface_asy << "xlimits(p2, xmin, xmax, true);"<< endl;
 output_interface_asy << "ylimits(p2, ymin, ymax, true);"<< endl;
 
 snprintf(st,sizeof(st),"%s",xlabel);
 output_interface_asy << "xaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),BottomTop,LeftTicks(Label(fontsize(30)),new real[]{" << x_min <<"," << x_max <<"})); "<< endl;
 snprintf(st,sizeof(st),"%s",ylabel);
 output_interface_asy << "yaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),LeftRight,RightTicks(Label(fontsize(30)),new real[]{"<< y_min << "," << (y_max+y_min)*0.5 << "," << y_max <<"})); "<< endl;
 
 output_interface_asy << "scale(Linear(true),Linear(true)); "<< endl;
 output_interface_asy << "add(p2.fit(250,250),(0,0),S);"<< endl;

if(q_production_run==0){
snprintf(st,sizeof(st),"/usr/local/texlive/2017/bin/x86_64-darwin/asy report_%s.asy",varname);
system(st);

snprintf(st,sizeof(st),"open report_%s.eps",varname);
system(st);
}
 
 
}


void plot_individual_function(const char *varname, const char *xlabel, int count1, double x_min, double x_max, const char *ylabel, double y_min, double y_max){

//snprintf(st,sizeof(st),"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/config.asy ."); system(st);
//snprintf(st,sizeof(st),"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/graph_routines.asy ."); system(st);


const char *varname1="potential_pic_individual";
const char *varname1a="pos_time_individual";
const char *varname1b="boltz_pic_individual";

const char *varname2="vel_time_individual";
const char *varname2b="vel_norm";

 //output file
 snprintf(st,sizeof(st),"report_%s_%d.asy",varname,count1);
 ofstream output_interface_asy(st,ios::out);
 
 //write output file
 output_interface_asy << "import graph;" << endl;
 output_interface_asy << "import stats;"<< endl;
 
 output_interface_asy << "from \"graph_routines.asy\" access *;"<< endl;
 
 output_interface_asy << "picture p2;"<< endl;
 output_interface_asy << "defaultpen(1.2);"<< endl;
 
 output_interface_asy << "real ymin=" << y_min << ";"<< endl;
 output_interface_asy << "real ymax=" << y_max << ";"<< endl;
 
 output_interface_asy << "real xmin=" << x_min << ";"<< endl;
 output_interface_asy << "real xmax=" << x_max << ";"<< endl;
 
 output_interface_asy << "size(p2,400,400,IgnoreAspect);"<< endl;
 output_interface_asy << "scale(p2,Linear(true),Linear(true));"<< endl;
 
 
if(varname==varname1){

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname1,count1,generation,0.1,0.1,0.1,1.1);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_scale_vertical(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g,%g);",varname1a,count1,generation,0.2,0.8,0.2,1.1,10.0);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_dashed_scale_vertical(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g,%g);",varname1b,count1,generation,0.0,0.0,0.0,0.5,10.0);
output_interface_asy << st << endl;

}

if(varname==varname2){

output_histogram_velocity_norm();

snprintf(st,sizeof(st),"simplot_simple(p2,\"report_%s_%d_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname2,count1,generation,0.2,0.8,0.2,1.1);
output_interface_asy << st << endl;

snprintf(st,sizeof(st),"simplot_simple_dashed_scale_vertical(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g,%g);",varname2b,generation,0.0,0.0,0.0,0.5,1.0);
output_interface_asy << st << endl;

}

 output_interface_asy << "xlimits(p2, xmin, xmax, true);"<< endl;
 output_interface_asy << "ylimits(p2, ymin, ymax, true);"<< endl;
 
 snprintf(st,sizeof(st),"%s",xlabel);
 output_interface_asy << "xaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),BottomTop,LeftTicks(Label(fontsize(30)),new real[]{" << x_min <<", 0, " << x_max <<"})); "<< endl;
 snprintf(st,sizeof(st),"%s",ylabel);
 output_interface_asy << "yaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),LeftRight,RightTicks(Label(fontsize(30)),new real[]{"<< y_min << "," << y_max <<"})); "<< endl;
 
 
 output_interface_asy << "scale(Linear(true),Linear(true)); "<< endl;
 output_interface_asy << "add(p2.fit(300,225),(0,0),S);"<< endl;

if(potential_pics<30){
if(q_production_run==0){
snprintf(st,sizeof(st),"/usr/local/texlive/2017/bin/x86_64-darwin/asy report_%s_%d.asy",varname,count1);
system(st);

snprintf(st,sizeof(st),"epstopdf report_%s_%d.eps",varname,count1);
system(st);

//snprintf(st,sizeof(st),"open report_%s_%d.eps",varname,count1);
//system(st);


if((count1==potential_pics) && (varname==varname1)){

char st2[1024] = "gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=report_combined.pdf";

    for(int i = 0; i < potential_pics+1; i++) {
        char temp[128];
        snprintf(temp, sizeof(temp), " report_potential_pic_individual_%d.pdf", i);
        strcat(st2, temp);
    }

    printf("%s\n", st2);
    system(st2);
    
   snprintf(st, sizeof(st), "open report_combined.pdf");
   system(st);
   
}

if((count1==potential_pics) && (varname==varname2)){

char st2[1024] = "gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=report_combined2.pdf";

    for(int i = 0; i < potential_pics+1; i++) {
        char temp[128];
        snprintf(temp, sizeof(temp), " report_vel_time_individual_%d.pdf", i);
        strcat(st2, temp);
    }

    printf("%s\n", st2);
    system(st2);
    
   snprintf(st, sizeof(st), "open report_combined2.pdf");
   system(st);
   
}
}


}
 
 
}



void read_net(void){

int i;

snprintf(st,sizeof(st),"net_in_gen_%d.dat",generation);
ifstream infile(st, ios::in);

for(i=0;i<number_of_net_parameters;i++){infile >> net_parameters[i];}

}

void output_net(void){

int i;

//parameter file
snprintf(st,sizeof(st),"net_out_gen_%d.dat",generation);
ofstream out_net(st,ios::out);

for(i=0;i<number_of_net_parameters;i++){out_net << net_parameters[i] << " ";}

}


void store_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters_holder[i]=net_parameters[i];}

}

void mutate_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){mutation[i]=mean_mutation[i]+gauss_rv(sigma_mutate);}
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]+=mutation[i];}

}


void restore_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]=net_parameters_holder[i];}

}

void initialize_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]=gauss_rv(0.0);}

}


void run_net(int net_id){

//inputs fixed externally and outputs used externally

int i,j,k;

//param offset
int pid=0;

double mu=0.0;
double sigma=0.0;
double delta=1e-4;

double hidden_node[width][depth];
double hidden_node_final[width_final];
const int number_of_inputs=number_inputs[net_id];
const int number_of_outputs=number_outputs[net_id];

//surface layer
for(i=0;i<width;i++){
hidden_node[i][0]=net_parameters[pid];pid++;
for(j=0;j<number_of_inputs;j++){hidden_node[i][0]+=inputs[j]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
mu=0.0;sigma=0.0;
for(i=0;i<width;i++){mu+=hidden_node[i][0];sigma+=hidden_node[i][0]*hidden_node[i][0];}
mu=mu/width;sigma=sigma/width;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width;i++){hidden_node[i][0]=(hidden_node[i][0]-mu)/sigma;}


//activation
for(i=0;i<width;i++){hidden_node[i][0]=tanh(hidden_node[i][0]);}

//stem layers
for(j=1;j<depth;j++){
for(i=0;i<width;i++){
hidden_node[i][j]=net_parameters[pid];pid++;
for(k=0;k<width;k++){hidden_node[i][j]+=hidden_node[k][j-1]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
mu=0.0;sigma=0.0;
for(i=0;i<width;i++){mu+=hidden_node[i][j];sigma+=hidden_node[i][j]*hidden_node[i][j];}
mu=mu/width;sigma=sigma/width;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width;i++){hidden_node[i][j]=(hidden_node[i][j]-mu)/sigma;}


//activation
for(i=0;i<width;i++){hidden_node[i][j]=tanh(hidden_node[i][j]);}
}

//final layer
for(i=0;i<width_final;i++){
hidden_node_final[i]=net_parameters[pid];pid++;
for(j=0;j<width;j++){hidden_node_final[i]+=hidden_node[j][depth-1]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
mu=0.0;sigma=0.0;
for(i=0;i<width_final;i++){mu+=hidden_node_final[i];sigma+=hidden_node_final[i]*hidden_node_final[i];}
mu=mu/width_final;sigma=sigma/width_final;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width_final;i++){hidden_node_final[i]=(hidden_node_final[i]-mu)/sigma;}


//activation
for(i=0;i<width_final;i++){hidden_node_final[i]=tanh(hidden_node_final[i]);}

//outputs
for(i=0;i<number_of_outputs;i++){
outputs[i]=net_parameters[pid];pid++;
for(j=0;j<width_final;j++){outputs[i]+=hidden_node_final[j]*net_parameters[pid];pid++;}
}

}


void jobcomplete(int i){

 //snprintf(st,sizeof(st),"rm jobcomplete.dat");
 //system(st);
 
 snprintf(st,sizeof(st),"jobcomplete.dat");
 ofstream output_job(st,ios::out);
 output_job << i << endl;
 output_job.close();

}


void run_trajectory_average(void){

int ok;
int cs;
int i,j;

reset_registers();

for(i=0;i<number_of_trajectories;i++){

cs=0;
ok=1;

for(j=0;j<n_reps;j++){

run_trajectory(j);

if(ok==1){
if((position<0) && (erasure_type==0)){cs++;success[j]++;}
if((position>=0) && (erasure_type==0)){ok=0;}
if((position<0) && (erasure_type==1)){ok=0;}
if((position>=0) && (erasure_type==1)){cs++;success[j]++;}
}

//work stats taken for each erasure, consistent with experment
wd[i*n_reps+j]=work;
work=0.0;

}

mean_cs+=cs;
if(q_production_run==0){cout << i << " " <<  mean_cs/(1.0*i+1.0) << endl;}


he[i]=heat/(1.0*n_reps);
pos[i]=position;
vel[i]=velocity;
teff[i]=velocity*velocity/(omega_0*omega_0);
etot[i]=potential()+0.5*velocity*velocity/(omega_0*omega_0);

}

for(i=0;i<n_reps;i++){success[i]=success[i]/(1.0*number_of_trajectories);}
mean_cs=mean_cs/(1.0*number_of_trajectories);

snprintf(st,sizeof(st),"report_rep_success_gen_%d.dat",generation);
ofstream out1(st,ios::out);
for(i=0;i<n_reps;i++){out1 << i << " " << success[i] << endl;}

//averaging (sets phi)
averaging();
output_trajectory_average_data();
traj_number=0;

}


void output_generational_data(int gen1){
if(gen1 % 100==0){

snprintf(st,sizeof(st),"report_phi.dat");
ofstream out1(st,ios::app);

snprintf(st,sizeof(st),"report_mean_work.dat");
ofstream out2(st,ios::app);

snprintf(st,sizeof(st),"report_mean_heat.dat");
ofstream out3(st,ios::app);

snprintf(st,sizeof(st),"report_mean_teff.dat");
ofstream out4(st,ios::app);

snprintf(st,sizeof(st),"report_mean_prob_erasure.dat");
ofstream out5(st,ios::app);

snprintf(st,sizeof(st),"report_mean_delta_position.dat");
ofstream out6(st,ios::app);

out1 << generation << " " << phi << endl;
out2 << generation << " " << mean_work << endl;
out3 << generation << " " << mean_heat << endl;
out4 << generation << " " << mean_teff << endl;
out5 << generation << " " << mean_prob_erasure << endl;
out6 << generation << " " << mean_delta_position << endl;

}}


void output_histogram(void){

snprintf(st,sizeof(st),"report_wd_gen_%d.dat",generation);
ofstream output_wd(st,ios::out);

snprintf(st,sizeof(st),"report_wd_weighted_gen_%d.dat",generation);
ofstream output_wd_weighted(st,ios::out);

int i;
int nbin=0;
const int bins=50;

double histo[bins];
double maxwork=0.0;
double minwork=0.0;

//work recorded elswehere
for(i=0;i<bins;i++){histo[i]=0.0;}

for(i=0;i<number_of_trajectories_histogram*n_reps;i++){

if(i==0){maxwork=wd[i];minwork=wd[i];}
else{

if(wd[i]>maxwork){maxwork=wd[i];}
if(wd[i]<minwork){minwork=wd[i];}
}}

//record width
histo_width=(maxwork-minwork)/(1.0*bins);

//safeguard
if((fabs(maxwork)<1e-6) && (fabs(minwork)<1e-6)){

histo_width=0.1;
maxwork=1.0;
minwork=0.0;

}

for(i=0;i<number_of_trajectories_histogram*n_reps;i++){

nbin=(int) (1.0*bins*(wd[i]-minwork)/(maxwork-minwork));
if(nbin==bins){nbin--;}

histo[nbin]+=1.0/(1.0*number_of_trajectories_histogram*n_reps);

}

//output
double w1;
for(i=0;i<bins;i++){

if(histo[i]>1e-6){
w1=maxwork*i/(1.0*bins)+minwork*(1.0-i/(1.0*bins))+0.5*(maxwork-minwork)/(1.0*bins);

output_wd << w1 << " " << histo[i]/histo_width << endl;
output_wd_weighted << w1 << " " << exp(-w1)*histo[i] << endl;


}
}

//plot histogram
plot_function("wd","W",-maxwork,maxwork,"P(W)",0,5.0/(histo_width*bins));

}


void ga(void){

int i;

record_trajectory=0;

//mutate net
if(q_production_run==1){mutate_net();}

//make protocol lookup table
make_lookup_table();

//calculate order parameter
run_trajectory_average();

//output evolutionary order parameter
snprintf(st,sizeof(st),"report_phi_gen_%d.dat",generation);
ofstream output_phi(st,ios::out);
output_phi << np << endl;

//output other order parameters
snprintf(st,sizeof(st),"report_order_parameters_gen_%d.dat",generation);
ofstream output_op(st,ios::out);
output_op << mean_prob_erasure << " " << mean_delta_position << " " << mean_work << " " << mean_heat << " " << mean_teff << " " << mean_total_energy << " " << mean_cs << endl;

//output histograms
output_histogram();
for(int i=0;i<potential_pics+2;i++){output_histogram_position(i);output_histogram_velocity(i);}

//output net
output_net();

//trajectory data
record_trajectory=1;

for(i=0;i<n_reps;i++){

set_erasure_type(i);
run_trajectory(i);

}

//jobcomplete
jobcomplete(1);

if(q_production_run==0){

//plots
plot_function("cee","t",-0.1,n_reps+0.1,"c",-1,8.5);
//plot_function("position","t",0,1,"x",-2.0*c1_boundary,2.0*c1_boundary);
plot_function("energy","t",-0.1,n_reps+0.1,"E",0,5);


if(n_reps==1){
plot_function("potential_by_well","t/t_{\\rm f}",-0.1,n_reps+0.1,"U",0,5);
plot_function("teff_by_well","t/t_{\\rm f}",-0.1,n_reps+0.1,"T_{\\rm eff}",0,5);
plot_function("work_by_well","t/t_{\\rm f}",-0.1,n_reps+0.1,"W",-2,7);
}
else{
plot_function("rep_position","n",0,n_reps,"x",-7,7);
plot_function("rep_teff","n",0,n_reps,"",0.9,3);
plot_function("rep_success","n",0,n_reps,"",0,1.05);
}

for(int i=0;i<potential_pics+1;i++){plot_individual_function("vel_time_individual","v",i,-3.0*omega_0,3.0*omega_0,"P(v)",0,65);}
for(int i=0;i<potential_pics+1;i++){plot_individual_function("potential_pic_individual","x",i,-2.0*c1_boundary,2.0*c1_boundary,"",-0.5,2.0*c1_boundary);}


}

}

void averaging(void){

int i,j;

//normalization
double n1=1.0/(1.0*number_of_trajectories);
double en[2]={1.0/(1.0*(number_of_trajectories-number_state_one)),1.0/(1.0*number_state_one)};

//reset counters
mean_teff=0.0;
mean_work=0.0;
mean_heat=0.0;
mean_total_energy=0.0;
mean_prob_erasure=0.0;
mean_delta_position=0.0;

for(i=0;i<number_of_trajectories;i++){

mean_work+=wd[i]*n1;
mean_heat+=he[i]*n1;
mean_teff+=teff[i]*n1;
mean_total_energy+=etot[i]*n1;
mean_delta_position+=(pos[i]+c1_boundary)*(pos[i]+c1_boundary)*n1;
if(pos[i]<0.0){mean_prob_erasure+=n1;}

}

mean_delta_position=sqrt(mean_delta_position);

//time-dependent averages
for(i=0;i<number_of_report_steps*n_reps;i++){

total_energy[i]*=n1;
kinetic_energy[i]*=n1;
potential_energy[i]*=n1;

for(j=0;j<2;j++){

teff_time[i][j]*=en[j];
work_time[i][j]*=en[j];
energy_time[i][j]*=en[j];
position_time[i][j]*=en[j];
velocity_time[i][j]*=en[j];

}}

//new phi
if(n_reps>1){np=1.0/(1.0*mean_cs);}
else{

if(mean_prob_erasure<0.9){np=mean_delta_position+50.0;}
else{np=1.0-mean_prob_erasure+0.01*mean_total_energy;}

}

}

double potential(void){

double p1=position-sign(position-c0)*c1;
double q1=0.5*kay*p1*p1;

//origin and continutity terms
q1+=2.0*kay*c0*c1*heaviside(position-c0);
q1-=2.0*kay*c0*c1*(1.0-heaviside(c0));

//double p1=position-c0;
//double q1=0.5*kay*p1*p1;

return (q1);

}


void initial_potential(void){

c0=c0_boundary;
c1=c1_boundary;

}

void final_potential(void){

c0=c0_boundary;
c1=c1_boundary;

}

void equilibrate(void){

//P_ex(x,v) = PDF(x) PDF(v)
//PDF(x) Gaussian with variance kT/k = nm
//PDF(v) Gaussian with variance kT/m = omega_0^2 (nm)^2

if((drand48()<0.5) || (record_trajectory==1)){initial_state=0;}
else{initial_state=1;}

//for averages
number_state_one+=initial_state;

//initialize position
position=(2.0*initial_state-1.0)*c1_boundary+gauss_rv(1.0/sqrt(kay));
//position=gauss_rv(1.0);

//initialize velocity
velocity=gauss_rv(omega_0);

//reset counters
tau=0.0;
work=0.0;
heat=0.0;
potential_pic_counter=0;

initial_potential();

}

void output_potential(int step_number){
if(record_trajectory==1){

int ok=0;
if(step_number % (trajectory_steps/potential_pics) == 0){ok=1;}
if(step_number==0){ok=1;}
if(step_number==trajectory_steps){ok=1;}

if(ok==1){

snprintf(st,sizeof(st),"report_boltz_pic_individual_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_boltz_individual(st,ios::out);

snprintf(st,sizeof(st),"report_potential_pic_individual_%d_gen_%d.dat",potential_pic_counter,generation);
ofstream out_pic_individual(st,ios::out);

int i;
const int n_points=2000;

double e1;
double e_min=0;
double e_values[n_points];

double x1=-2.0*c1_boundary;
double x2=2.0*c1_boundary;
double zed=0.0;
double delta_x=0.0;


//record position
double position_holder=position;

for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
e1=potential();

out_pic_individual << position << " " << e1 << endl;

}

//boltzmann weight
//log energies; compute minimum
delta_x=(x2-x1)/(1.0*n_points);
for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
e1=potential();

if(i==0){e_min=e1;}
else{if(e1<e_min){e_min=e1;}}

e_values[i]=e1;

}

//calculate Z
for(i=0;i<n_points;i++){

e_values[i]-=e_min;
zed+=delta_x*exp(-e_values[i]);

}

//plot point
for(i=0;i<n_points;i++){

position=x1+(x2-x1)*i/(1.0*n_points-1.0);
out_boltz_individual << position << " " << exp(-e_values[i])/zed << endl;

}

//reset position
position=position_holder;
potential_pic_counter++;

}}}



void output_histogram_position(int time_slice){

snprintf(st,sizeof(st),"report_pos_time_individual_%d_gen_%d.dat",time_slice,generation);
ofstream output_pos_time_individual(st,ios::out);

int i;
int nbin=0;
const int bins=50;

double histo[bins];
double maxpos=0.0;
double minpos=0.0;

//pos recorded elswehere
for(i=0;i<bins;i++){histo[i]=0.0;}

for(i=0;i<number_of_trajectories_histogram;i++){

if(i==0){maxpos=pos_time[i][time_slice];minpos=pos_time[i][time_slice];}
else{

if(pos_time[i][time_slice]>maxpos){maxpos=pos_time[i][time_slice];}
if(pos_time[i][time_slice]<minpos){minpos=pos_time[i][time_slice];}
}}

//record width
histo_width=(maxpos-minpos)/(1.0*bins);

//safeguard
if((fabs(maxpos)<1e-6) && (fabs(minpos)<1e-6)){

histo_width=0.1;
maxpos=1.0;
minpos=0.0;

}

for(i=0;i<number_of_trajectories_histogram;i++){

nbin=(int) (1.0*bins*(pos_time[i][time_slice]-minpos)/(maxpos-minpos));
if(nbin==bins){nbin--;}

histo[nbin]+=1.0/(1.0*number_of_trajectories_histogram);

}

//output
double x1;
for(i=0;i<bins;i++){

if(histo[nbin]>0.5/(1.0*number_of_trajectories_histogram)){
x1=maxpos*i/(1.0*bins)+minpos*(1.0-i/(1.0*bins))+0.5*(maxpos-minpos)/(1.0*bins);

output_pos_time_individual << x1 << " " << histo[i]/histo_width << endl;


}
}

//histogram plotted from position histogram

}


void output_histogram_velocity(int time_slice){

snprintf(st,sizeof(st),"report_vel_time_individual_%d_gen_%d.dat",time_slice,generation);
ofstream output_vel_time_individual(st,ios::out);

int i;
int nbin=0;
const int bins=50;

double histo[bins];
double maxvel=0.0;
double minvel=0.0;

//vel recorded elswehere
for(i=0;i<bins;i++){histo[i]=0.0;}

for(i=0;i<number_of_trajectories_histogram;i++){

if(i==0){maxvel=vel_time[i][time_slice];minvel=vel_time[i][time_slice];}
else{

if(vel_time[i][time_slice]>maxvel){maxvel=vel_time[i][time_slice];}
if(vel_time[i][time_slice]<minvel){minvel=vel_time[i][time_slice];}
}}

//record width
histo_width=(maxvel-minvel)/(1.0*bins);

//safeguard
if((fabs(maxvel)<1e-16) && (fabs(minvel)<1e-16)){

histo_width=0.1;
maxvel=1.0;
minvel=0.0;

}

for(i=0;i<number_of_trajectories_histogram;i++){

nbin=(int) (1.0*bins*(vel_time[i][time_slice]-minvel)/(maxvel-minvel));
if(nbin==bins){nbin--;}

histo[nbin]+=1.0/(1.0*number_of_trajectories_histogram);

}

//output
double x1;
for(i=0;i<bins;i++){

if(histo[nbin]>0.5/(1.0*number_of_trajectories_histogram)){
x1=maxvel*i/(1.0*bins)+minvel*(1.0-i/(1.0*bins))+0.5*(maxvel-minvel)/(1.0*bins);

output_vel_time_individual << x1 << " " << histo[i]/histo_width << endl;


}
}

//histogram plotted from velocity histogram

}


void record_position(int step_number){

int ok=0;
int entry=0;
int dt=trajectory_steps/potential_pics;

if(step_number==0){ok=1;}
if(step_number==trajectory_steps){ok=1;entry=potential_pics+1;}
if(step_number % dt == 0){ok=1;entry=step_number/dt;}

if(ok==1){pos_time[traj_number][entry]=position;}
if(ok==1){vel_time[traj_number][entry]=velocity;}
if(ok==1){te_time[traj_number][entry]=velocity*velocity/(omega_0*omega_0);}

}


void reset_registers(void){

int i,j;

traj_number=0;
number_state_one=0;

for(i=0;i<number_of_report_steps*n_reps;i++){

total_energy[i]=0.0;
kinetic_energy[i]=0.0;
potential_energy[i]=0.0;

for(j=0;j<2;j++){

teff_time[i][j]=0.0;
work_time[i][j]=0.0;
energy_time[i][j]=0.0;
position_time[i][j]=0.0;
velocity_time[i][j]=0.0;

}}

//<consecutive successes>
mean_cs=0.0;

}



void record_trajectory_averages(int step_number, int rep_number){

int s1;

if(step_number % report_step==0){

s1=step_number/report_step;
if(s1<number_of_report_steps){

s1+=number_of_report_steps*rep_number;

potential_energy[s1]+=potential();
kinetic_energy[s1]+=0.5*velocity*velocity/(omega_0*omega_0);
total_energy[s1]+=(potential()+0.5*velocity*velocity/(omega_0*omega_0));

work_time[s1][initial_state]+=work;
energy_time[s1][initial_state]+=potential();
position_time[s1][initial_state]+=position;
velocity_time[s1][initial_state]+=velocity;
teff_time[s1][initial_state]+=velocity*velocity/(omega_0*omega_0);

}}

}

void output_trajectory_average_data(void){

int i;
int nr;
double t1;

snprintf(st,sizeof(st),"report_work_average_state_0_gen_%d.dat",generation);
ofstream out1(st,ios::app);

snprintf(st,sizeof(st),"report_work_average_state_1_gen_%d.dat",generation);
ofstream out2(st,ios::app);

snprintf(st,sizeof(st),"report_energy_average_state_0_gen_%d.dat",generation);
ofstream out5(st,ios::app);

snprintf(st,sizeof(st),"report_energy_average_state_1_gen_%d.dat",generation);
ofstream out6(st,ios::app);

snprintf(st,sizeof(st),"report_position_average_state_0_gen_%d.dat",generation);
ofstream out9(st,ios::app);

snprintf(st,sizeof(st),"report_position_average_state_1_gen_%d.dat",generation);
ofstream out10(st,ios::app);

snprintf(st,sizeof(st),"report_teff_average_state_0_gen_%d.dat",generation);
ofstream out11(st,ios::app);

snprintf(st,sizeof(st),"report_teff_average_state_1_gen_%d.dat",generation);
ofstream out12(st,ios::app);

snprintf(st,sizeof(st),"report_total_energy_gen_%d.dat",generation);
ofstream out13(st,ios::app);

snprintf(st,sizeof(st),"report_kinetic_energy_gen_%d.dat",generation);
ofstream out14(st,ios::app);

snprintf(st,sizeof(st),"report_potential_energy_gen_%d.dat",generation);
ofstream out15(st,ios::app);

snprintf(st,sizeof(st),"report_rep_teff_gen_%d.dat",generation);
ofstream out16(st,ios::app);

for(i=0;i<number_of_report_steps*n_reps;i++){

t1=1.0*i/(1.0*number_of_report_steps);

out1 << t1 << " " << work_time[i][0] << endl;
out2 << t1 << " " << work_time[i][1] << endl;

out5 << t1 << " " << energy_time[i][0] << endl;
out6 << t1 << " " << energy_time[i][1] << endl;

out9 << t1 << " " << position_time[i][0] << endl;
out10 << t1 << " " << position_time[i][1] << endl;

out11 << t1 << " " << teff_time[i][0] << endl;
out12 << t1 << " " << teff_time[i][1] << endl;

out13 << t1 << " " << total_energy[i] << endl;
out14 << t1 << " " << kinetic_energy[i] << endl;
out15 << t1 << " " << potential_energy[i] << endl;

if((i>0) && (i % number_of_report_steps==0)){
nr=(int) (t1);out16 << nr << " " << 2.0*kinetic_energy[i] << endl;
}

}
}

void output_histogram_velocity_norm(void){

snprintf(st,sizeof(st),"report_vel_norm_gen_%d.dat",generation);
ofstream out1(st,ios::out);


int i;
int n_points=1000;

double q1;
double vee;
double v1=-3*omega_0;
double v2=3*omega_0;
double dv=(v2-v1)/(1.0*n_points-1.0);
double q_norm=1.0/(sqrt(2.0*pi)*omega_0);


for(i=0;i<n_points;i++){

vee=v1+i*dv;
q1=q_norm*exp(-vee*vee/(2.0*omega_0*omega_0));

out1 << vee << " " << q1 << endl;

}

}


void default_protocol(void){

if(tau<tau_protocol){

//default protocol
if(tau/tau_protocol<0.5){
c0=c0_boundary;
c1=c1_boundary*(1.0-2.0*tau/tau_protocol);
}
else{
c0=8.0;
c1=2.0*c1_boundary*(tau/tau_protocol-0.5);
}

}
else{
//quiescent period

c0=c0_boundary;
c1=c1_boundary;

}

}


void net_protocol(void){

if(tau<tau_protocol){inputs[0]=tau/tau_protocol;run_net(0);}
else{outputs[0]=0.0;outputs[1]=0.0;}

c0=c0_boundary+outputs[0];
c1=c1_boundary+outputs[1];

}




int sign(double x){
    return (x > 0.0) - (x < 0.0);
}

double heaviside(double x){

if(x<0.0){return (0.0);}
else{
if(x>0.0){return (1.0);}
else{return (0.5);}
}

}


void make_lookup_table(void){

if(q_protocol==1){read_semi_optimal_protocol();}
else{

int i;

tau=0.0;
for(i=0;i<trajectory_steps;i++){

//protocol
if(q_protocol==0){default_protocol();}
if(q_protocol==2){net_protocol();}

//set lookup table
lookup_table[i][0]=c0;
lookup_table[i][1]=c1;

//advance time
tau+=1.0/(1.0*trajectory_steps);

}}

//zero time
tau=0.0;

}


void shift_velocity(double dv){

work-=0.5*velocity*velocity/(omega_0*omega_0);

velocity+=dv;

work+=0.5*velocity*velocity/(omega_0*omega_0);

}

void shift_c1(double dv){

c1+=dv/(omega_0*omega_0*timestep);

//c1+=dv;


}


void set_erasure_type(int i){

if(i==0){erasure_type=0;}
else{

if(drand48()<0.5){erasure_type=0;}
else{erasure_type=1;}

}
}

void read_semi_optimal_protocol(void){

int i;

double dt;
double dt_min;
double t1,x0,x1;

tau=0.0;
for(i=0;i<trajectory_steps;i++){

if(tau<=0.5){
//read from protocol

dt_min=100;

ifstream infile0("semi_optimal.dat", ios::in);
while (!infile0.eof ()){

infile0 >> t1 >> x0 >> x1;

t1*=0.5;
dt=fabs(t1-tau);
if(dt<dt_min){dt_min=dt;c0=x0;c1=x1;}

}}
else{c0=0.0;c1=5.0;}

//set lookup table
lookup_table[i][0]=c0;
lookup_table[i][1]=c1;


//advance time
tau+=1.0/(1.0*trajectory_steps);

}
}
