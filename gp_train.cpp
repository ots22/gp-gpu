#include <libconfig.h++>
#include "gp.hpp"
#include "optim.hpp"

#define QUOTE(x) #x
#define STR(x) QUOTE(x)
#ifdef LIBCONFIG_VER_MAJOR
#define MY_LIBCONFIG_VER_STRING STR(LIBCONFIG_VER_MAJOR) "." STR(LIBCONFIG_VER_MINOR) "." STR(LIBCONFIG_VER_REVISION)
#else
#define MY_LIBCONFIG_VER_STRING "unknown"
#endif

using namespace std;
using namespace libconfig;

int main(void)
{
	unsigned N(0), Ninput(0), Nhypers(0); // number of input records (N), length of input vector (Ninput) and number of hyperparameters (Nhypers)
	bool optim_p(false); // perform a maximum liklihood optimization?
	string outfile("out.gp"); // filename to output the GP

	// read a config file for the iteration
	Config cfg;
	try {
		cfg.readFile("gp_train.cfg");
	}
	catch (ParseException &pe) {
		cerr << "There was an error reading the configuration file, gp_train.cfg.\n"
		     << "On line " << pe.getLine() <<  ": " << pe.getError() << ".\n"
		     << "Please consult the documentation for libconfig++ (version " << MY_LIBCONFIG_VER_STRING << ").\n"
		     << "Terminating.\n";
		exit(1);
	}
	cfg.lookupValue("N",N);
	cfg.lookupValue("Ninput", Ninput);
	cfg.lookupValue("Nhypers", Nhypers);
	cfg.lookupValue("optimize", optim_p);
	cfg.lookupValue("outfile", outfile);

	vec hypers(Nhypers), lbound(Nhypers), ubound(Nhypers);
	hypers.fill(0.0); lbound.fill(0.0); ubound.fill(0.0);
	cfg.lookup("hypers");
	Setting &hypers_cfg  = cfg.lookup("hypers");
	for (int i=0; i<Nhypers; i++) hypers(i) = hypers_cfg[i];

	if (optim_p) {
		Setting &lbound_cfg = cfg.lookup("lbound");
		for (int i=0; i<Nhypers; i++) lbound(i) = lbound_cfg[i];
		Setting &ubound_cfg = cfg.lookup("ubound");
		for (int i=0; i<Nhypers; i++) ubound(i) = ubound_cfg[i];
	}

	cout << "Successfully read configuration file." << endl
	     << "N = "        << N       << "\n"
	     << "Ninput = "   << Ninput  << "\n"
	     << "Nhypers = "  << Nhypers << "\n"
	     << "optimize = " << std::boolalpha << optim_p << "\n"
	     << "hypers = \n" << hypers << "\n";
	if (optim_p) cout << "lbound = \n" << lbound << "\n"
			  << "ubound = \n" << ubound << endl;

	mat xs(N,Ninput);
	vec ts(N);
	ivec Ts(N);

	cout << "Reading the training data from stdin." << endl;
	for (unsigned lineno=0; lineno < N; lineno++) {
		for (unsigned i=0; i < Ninput; i++) cin >> xs(lineno,i);
		cin >> Ts(lineno);
		cin >> ts(lineno);
	}
	cout << "Constructing the GP." << endl;
	DenseGP gp(hypers, xs, ts, Ts);
	if (optim_p) {
		cout << "Commencing optimization." << endl;
		log_lik_optim(gp, lbound, ubound, 1000, 1e-6);
	}
	cout << "Saving GP to " << outfile << "." << endl;
	gp.save(outfile);
	cout << "Done!" << endl;
}
