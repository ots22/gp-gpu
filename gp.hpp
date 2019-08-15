#ifndef GP_HPP
#define GP_HPP

#include <armadillo>
#include <algorithm>
#include <string>
#include <assert.h>
#include "cov.hpp"

using arma::ivec;
using arma::vec;
using arma::Col;
using arma::rowvec;
using arma::mat;
using arma::Mat;
using arma::arma_ascii;
using arma::arma_binary;

typedef int obs_kind;
typedef ivec vec_obs_kind;


class GP {
	virtual void update_matrices(void)=0;

protected:
	// Up to derived classes and the covariance function how to interpret this
	// Stored in a single vector for NLOpt
	vec hypers;
	
public:
	void set_hypers(vec new_hypers)
	{
		if (!std::equal(new_hypers.begin(), new_hypers.end(), hypers.begin())) {
			std::copy(new_hypers.begin(), new_hypers.end(), hypers.begin());
			update_matrices();
		}
	}

	vec get_hypers(void) const
	{
		return hypers;
	}

	int Nhypers(void) const { return hypers.n_rows; }
	virtual int data_length(void) const = 0;

	virtual double log_lik(void) const = 0;
	virtual double predict(vec xnew, obs_kind T, vec& work) const = 0;

	virtual ~GP() {};
	GP(vec hypers) : hypers(hypers) {}
	GP(void) {};
};

class DenseGP : public GP {

	unsigned int N, Ninput;

	// training inputs (rows)
	mat xs;
	vec ts;
	vec_obs_kind Ts;

	// covariance matrix and its inverse
	mat C, invC;

	// precomputed product, used in the prediction
	vec invCts;

	// The component of this vector corresponds to an observation kind:
	// Noise on value observations component 0, noise on derivatives 1..Ninput (inclusive)
	double noise(void) const {
		return 4.0e-8;
	}
	
	// The remaining hyperparameters (e.g. lengthscales etc)
	vec theta(void) const {
		return hypers;
	}

	virtual void update_matrices(void) final
	{
		for (unsigned j=0; j<N; j++) {
			for (unsigned i=0; i<N; i++) {
				vec xi = xs.row(i).t();
				vec xj = xs.row(j).t();
				C(i,j) = cov(Ts(i), Ts(j), xi, xj, theta());
				if (i==j) C(i,j) += noise();
			}
		}
		invCts = solve(C,ts);
	}
public:
	virtual int data_length(void) const
	{
		return N;
	}

	virtual double log_lik(void) const final
	{
		double logdet;
		double sign;
		arma::log_det(logdet, sign, C);

		return -0.5 * (logdet + dot(ts, invCts));
	}

	virtual double predict(vec xnew, obs_kind Tnew, vec& work) const final
	{
		// work === k, of length N
		for (unsigned i=0; i<N; i++) {
			work(i) = cov(Tnew, Ts(i), xnew, xs.row(i).t(), theta());
		}

		return dot(work, invCts);
	}

	virtual double predict(vec xnew, obs_kind Tnew) const final
	{
		vec k(N);
		for (unsigned i=0; i<N; i++) {
			k(i) = cov(Tnew, Ts(i), xnew, xs.row(i).t(), theta());
		}

		return dot(k, invCts);
		
	}

	DenseGP(vec hypers, mat xs, vec ts, vec_obs_kind Ts)
		: GP(hypers),
		  N(xs.n_rows), Ninput(xs.n_cols),
		  xs(xs), ts(ts), Ts(Ts),
		  C(N,N), invC(N,N), invCts(N)
	{
		update_matrices();

		assert(ts.n_rows == N);
		assert(Ts.n_rows == N);
	};

	void print(void)
	{
		std::cout << N << " " << Ninput << std::endl;
		std::cout << "\nhypers\n" << hypers << "\n\nxs\n" << xs;
		std::cout << "\n\nts\n" << ts << "\n\nTs" << Ts;
		std::cout << "\n\nC\n" << C << "\n\ninvC\n" << invC;
		std::cout << "\n\ninvCts" << invCts << "\n";
	}
	
	virtual void save(std::string filename)
	{
		std::ofstream out(filename);
		out << N << "\n"
		    << Ninput << "\n";
		hypers.save(out, arma_binary);
		xs.save(out, arma_binary);
		ts.save(out, arma_binary);
		Ts.save(out, arma_binary);
		C.save(out, arma_binary);
		invC.save(out, arma_binary);
		invCts.save(out, arma_binary);
	}

	DenseGP(std::string filename)
	{
		std::ifstream in(filename);
		in >> N >> Ninput;
		hypers.load(in, arma_binary);
		xs.load(in, arma_binary);
		ts.load(in, arma_binary);
		Ts.load(in, arma_binary);
		C.load(in, arma_binary);
		invC.load(in, arma_binary);
		invCts.load(in, arma_binary);		
	}
};

#endif
