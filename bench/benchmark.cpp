#include "romenskii.hpp"
#include <chrono>
#include <random>
#include "gp_gpu.hpp"
class Timer {
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

int main(int argc, char *argv[])
{

	double copper_c0 = 4.6, copper_b0 = 2.1;
	double copper_K0 = copper_c0*copper_c0 - (4.0/3.0)*copper_b0*copper_b0,
		copper_B0 = copper_b0*copper_b0;

	double copper_rho0 = 8.93;

	romenskii::Romenskii eos(copper_rho0, copper_K0, copper_B0, 1.0, 3.0, 2.0, 3.9e-4, 300);

	int Npoints = 0;
	Npoints = atoi(argv[1]);
	const int Ninput = 7;
	
	std::default_random_engine rgen(101);
	std::uniform_real_distribution<double> G_diag_d(0.95, 1.05);
	std::uniform_real_distribution<double> G_off_diag_d(-0.05, 0.05);
	std::uniform_real_distribution<double> potT_d(0.0, 160.0);

	DenseGP gp_cpu("eos-example.gp");
	DenseGP_GPU gp_gpu("eos-example.gp");

	mat xs(Ninput, Npoints);
	// generate some random points
	for (int i=0; i<Npoints; i++) {
		xs.col(i) = vec{G_diag_d(rgen), G_diag_d(rgen), G_diag_d(rgen), G_off_diag_d(rgen), G_off_diag_d(rgen), G_off_diag_d(rgen), potT_d(rgen)};
	}
	Mat<REAL> Rxs = arma::conv_to<Mat<REAL> >::from(xs);
	
	
	vec result_cpu(Npoints);
	Col<REAL> result_gpu(Npoints), result_romenski(Npoints);

	Timer stopwatch;
	double duration_cpu, duration_gpu, duration_romenski;


	///////////// start clock /////////////
	stopwatch.reset();

#pragma omp parallel for	
	for (int i=0; i<Npoints; i++) {
		result_cpu(i) = gp_cpu.predict(xs.col(i), 0);
	}

	duration_cpu = stopwatch.elapsed();
	/////////////  stop clock  ////////////



      
	///////////// start clock /////////////
	stopwatch.reset();

	gp_gpu.predict_batch(result_gpu, Rxs, 0);

	duration_gpu = stopwatch.elapsed();
	/////////////  stop clock  ////////////




	// ///////////// start clock /////////////
	stopwatch.reset();

#pragma omp parallel for	
	for (int i=0; i<Npoints; i++) {
		arma::mat33 G;
		G(0,0) = xs(0,i); G(0,1) = xs(5,i); G(0,2) = xs(4,i);
		G(1,0) = xs(5,i); G(1,1) = xs(1,i); G(1,2) = xs(3,i);
		G(2,0) = xs(4,i); G(2,1) = xs(3,i); G(2,2) = xs(2,i);
		result_romenski(i) = eos.e_internal(xs(6,i),G);
	}

	duration_romenski = stopwatch.elapsed();
	// /////////////  stop clock  ////////////


	vec diff (result_gpu - result_cpu);
	double err = norm(diff,1.0)/Npoints;
	
	std::cout << "CPU: time for " << Npoints << " evaluations was " << duration_cpu << " seconds\n";
	std::cout << "GPU: time for " << Npoints << " evaluations was " << duration_gpu << " seconds\n";
	std::cout << "mean absolute difference between the results was " << err << std::endl;
	std::cout << "(CPU: time for the analytical expression was " << duration_romenski << " seconds)\n";

	//std::cout << Npoints << " " << 1000*duration_cpu << " " << 1000*duration_gpu << std::endl;
}
