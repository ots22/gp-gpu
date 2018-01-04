#include <chrono>
#include <random>
#include "gp.hpp"
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

int main(void)
{
	const int Npoints = 20000;
	const int Ninput = 7;
	
	std::default_random_engine rgen(101);
	std::uniform_real_distribution<double> G_diag_d(0.95, 1.05);
	std::uniform_real_distribution<double> G_off_diag_d(-0.05, 0.05);
	std::uniform_real_distribution<double> potT_d(100.0, 800.0);

	DenseGP gp("eos-example.gp");

	mat xs(Ninput, Npoints);
	// generate some random points
	for (int i=0; i<Npoints; i++) {
		xs.col(i) = vec{G_diag_d(rgen), G_diag_d(rgen), G_diag_d(rgen), G_off_diag_d(rgen), G_off_diag_d(rgen), G_off_diag_d(rgen), potT_d(rgen)};
	}
	
	vec result(Npoints);


	Timer stopwatch;
	double duration;
	///////////// start clock /////////////
	stopwatch.reset();

	for (int i=0; i<Npoints; i++) {
		result(i) = gp.predict(xs.col(i), 0);
	}

	duration = stopwatch.elapsed();
	/////////////  stop clock  ////////////
	std::cout << "CPU, time for " << Npoints << " predictions was " << duration << std::endl;
	std::cout << "One of the results was " << result(19450) << std::endl;

}
