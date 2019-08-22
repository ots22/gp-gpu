#include <libconfig.h++>
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

void read_cfg(vec &lbound, vec &ubound)
{
	const int Ninput = lbound.n_rows;
	using namespace libconfig;
	Config cfg;
	cfg.readFile("benchmark.cfg");
	
	Setting &lbound_cfg = cfg.lookup("lbound");
	for (int i=0; i < Ninput; i++) lbound(i) = lbound_cfg[i];
	Setting &ubound_cfg = cfg.lookup("ubound");
	for (int i=0; i < Ninput; i++) ubound(i) = ubound_cfg[i];	
}

int main(int argc, char *argv[])
{
	int Npoints = 0;
	Npoints = atoi(argv[1]);
	
	DenseGP gp_cpu("eos-example.gp");
	DenseGP_GPU gp_gpu("eos-example.gp");

	int Ninput = gp_cpu.input_dimension();
	vec lbound(Ninput);
	vec ubound(Ninput);

	try {
		read_cfg(lbound, ubound);
	} catch  (libconfig::ParseException &pe) {
		std::cerr << "There was an error reading the configuration file, benchmark.cfg.\n"
			  << "On line " <<pe.getLine() << ": " << pe.getError() << ".\n"
			  << "Terminating.\n";
		exit(1);
	}

	// generate some random points
	mat xs(Ninput, Npoints, arma::fill::randu);

	// scale and shift
	xs.each_col() %= (ubound - lbound);
	xs.each_col() += lbound;

	Mat<REAL> Rxs = arma::conv_to<Mat<REAL> >::from(xs);
	
	
	vec result_cpu(Npoints);
	vec result_cpu_var(Npoints);
	Col<REAL> result_gpu(Npoints);
	Col<REAL> result_gpu_var(Npoints);

	Timer stopwatch;
	double duration_cpu, duration_cpu_var, duration_gpu, duration_gpu_var;


	///////////// start clock /////////////
	stopwatch.reset();

#       pragma omp parallel for	
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


	///////////// start clock /////////////
	stopwatch.reset();

#       pragma omp parallel for	
	for (int i=0; i<Npoints; i++) {
		result_cpu(i) = gp_cpu.predict_variance(xs.col(i), 0, result_cpu_var(i));
	}

	duration_cpu_var = stopwatch.elapsed();
	/////////////  stop clock  ////////////



	
	///////////// start clock /////////////
	stopwatch.reset();

	gp_gpu.predict_batch_variance(result_gpu, result_gpu_var, Rxs, 0);

	duration_gpu_var = stopwatch.elapsed();
	/////////////  stop clock  ////////////

	vec diff_var (result_cpu_var - result_gpu_var);
	double err_var = norm(diff_var,1.0)/Npoints;

	
	// vec trial{0.94785, 1.11898, 1.09813, -0.230669, -0.141593, -0.026315, 53.6836};
	// double expected(6.93661);
	// double var;
	// std::cout << gp_cpu.predict_variance(trial, 0, var) << std::endl;
	// std::cout << "   (predicted std. dev. of " << sqrt(var) << " )" << std::endl;
	// std::cout << gp_gpu.predict(trial, 0) << std::endl;
	// std::cout << expected << std::endl;

	// vec trial{2.5};
	// double expected(2.5 * 2.5);
	// double var;
	// std::cout << gp_cpu.predict_variance(trial, 0, var) << std::endl;
	// std::cout << "   (predicted std. dev. of " << sqrt(var) << " )" << std::endl;
	// std::cout << gp_gpu.predict(trial, 0) << std::endl;
	// std::cout << expected << std::endl;

	
	// vec mean(3);
	// vec variance(3);
	
	// gp_gpu.predict_batch_variance(mean, variance, mat{1.5, 2.5, 3.5}, 0);
	
	// for (int i = 0; i < 1000; i++) {
	// 	double var;
	// 	std::cerr << 0.01 * i << "  "
	// 		  << gp_cpu.predict_variance(vec{0.01 * i}, 0, var) << " " << var << " ";
	// 	std::cerr << gp_gpu.predict_variance(vec{0.01 * i}, 0, var) << " " << var << "\n";
	// }

	// std::cout << "mean: " << mean << "var: " << variance << "\n";
	
	std::cout << "CPU: time for " << Npoints << " evaluations was:\n"
		  << "    mean only:          " << duration_cpu << " seconds\n"
		  << "    including variance: " << duration_cpu_var << " seconds\n\n";

	std::cout << "GPU: time for " << Npoints << " evaluations was:\n"
		  << "    mean only:          " << duration_gpu << " seconds\n"
		  << "    including variance: " << duration_gpu_var << " seconds\n\n";
	
	std::cout << "mean absolute difference between predicted means: " << err << std::endl;
	std::cout << "mean absolute difference between predicted variance: " << err_var << std::endl;
	
}
