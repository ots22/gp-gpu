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
	int Npoints = 0;
	Npoints = atoi(argv[1]);
	const int Ninput = 7;
	
	std::default_random_engine rgen(101);
	std::uniform_real_distribution<double> G_diag_d(0.95, 1.05);
	std::uniform_real_distribution<double> G_off_diag_d(-0.05, 0.05);
	std::uniform_real_distribution<double> potT_d(0.0, 150.0);

	DenseGP gp_cpu("eos-example.gp");
	DenseGP_GPU gp_gpu("eos-example.gp");

	mat xs(Ninput, Npoints);
	// generate some random points
	for (int i=0; i<Npoints; i++) {
		xs.col(i) = vec{G_diag_d(rgen), G_diag_d(rgen), G_diag_d(rgen), G_off_diag_d(rgen), G_off_diag_d(rgen), G_off_diag_d(rgen), potT_d(rgen)};
	}
	Mat<REAL> Rxs = arma::conv_to<Mat<REAL> >::from(xs);
	
	
	vec result_cpu(Npoints);
	Col<REAL> result_gpu(Npoints);

	Timer stopwatch;
	double duration_cpu, duration_gpu;


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

	vec diff (result_gpu - result_cpu);
	double err = norm(diff,1.0)/Npoints;

	vec trial{0.94785, 1.11898, 1.09813, -0.230669, -0.141593, -0.026315, 53.6836};
	double expected(6.93661);
	//std::cout << gp_cpu.predict(trial, 0) << std::endl;
	//std::cout << gp_gpu.predict(trial, 0) << std::endl;
	//std::cout << expected << std::endl;
	
	std::cout << "CPU: time for " << Npoints << " evaluations was " << duration_cpu << " seconds\n";
	std::cout << "GPU: time for " << Npoints << " evaluations was " << duration_gpu << " seconds\n";
	std::cout << "mean absolute difference between the results was " << err << std::endl;
}
