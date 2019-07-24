#ifndef ROMENSKII_HPP
#define ROMENSKII_HPP

#include "eos.hpp"

namespace romenskii {

class Romenskii : public Eos {
	const double rho_0, K0, B0, alpha, beta, gamma, cv, T0;
public:
	virtual double rho0(void) const { return rho_0; }
	
	virtual double e_internal(double S, arma::mat33 G) const
	{
		assert(is_symmetric(G));
		double G11=G(0,0), G12=G(0,1), G13=G(0,2), G22=G(1,1), G23=G(1,2), G33=G(2,2);
		double result = -99999;
#               include "sympy/romenskii_energy.inc"
		return result;
	}
	
	virtual double entropy(double E, arma::mat33 G) const
	{
		assert(is_symmetric(G));
		double G11=G(0,0), G12=G(0,1), G13=G(0,2), G22=G(1,1), G23=G(1,2), G33=G(2,2);
		double result = -99999;
#               include "sympy/romenskii_entropy.inc"
		return result;
	}

	virtual arma::mat33 sigma(double S, arma::mat33 G) const
	{
		assert(is_symmetric(G));

		double G11=G(0,0), G12=G(0,1), G13=G(0,2), G22=G(1,1), G23=G(1,2), G33=G(2,2);
		arma::mat33 result;
		result.fill(-99999.0);
#               include "sympy/romenskii_stress.inc"

		assert(is_approx_symmetric(result));
				
		return result;
	}

	// Useful helper routine for the F derivatives, since stress
	// is expressed in terms of G rather than F.
 	arma::mat33 dsigma_dG(double S, arma::mat33 G, int i, int j) const
 	{
		double G11=G(0,0), G12=G(0,1), G13=G(0,2), G22=G(1,1), G23=G(1,2), G33=G(2,2);
		arma::mat33 result;
		result.fill(-99999);
 		switch (i + 10*j) {
 		case 00:
#                       include "sympy/romenskii_dsigma11_dG.inc"
 			break;
		case 01:
#                       include "sympy/romenskii_dsigma12_dG.inc"
			break;
		case 02:
#			include "sympy/romenskii_dsigma13_dG.inc"
			break;
 		case 10:
#			include "sympy/romenskii_dsigma21_dG.inc"
			break;
		case 11:
#                       include "sympy/romenskii_dsigma22_dG.inc"
 			break;
		case 12:
#                       include "sympy/romenskii_dsigma23_dG.inc"
			break;
		case 20:
#			include "sympy/romenskii_dsigma31_dG.inc"
			break;
		case 21:
#                       include "sympy/romenskii_dsigma32_dG.inc"
			break;
		case 22:
#                       include "sympy/romenskii_dsigma33_dG.inc"
 			break;		
 		default:
 			std::cerr << "Invalid case in " << __PRETTY_FUNCTION__ << std::endl;
 			exit(1);
		}
		return result;
	}

  	virtual arma::mat33 A(double S, arma::mat33 F, int n, int beta) const
 	{
		arma::mat33 inv_F(inv(F));
		arma::mat33 G(inv_F * trans(inv_F));

		double rho = rho_0 / det(F);

  		arma::mat33 result;
		result.fill(0.0);
		for (int i=0; i<3; i++)
			for (int j=0; j<3; j++)
				// reduce over both components of G:
				result(i,j) = trace(dG_dF(beta, j, inv_F)
						    * trans(dsigma_dG(S,G,n,i)));
// equivalent to the following...		
//				for (int p=0; p<3; p++)
//					for (int q=0; q<3; q++)
//						result(i,j) += 
//							dG_dF(beta, j, inv_F)(p,q) 
//							* dsigma_dG(S,G,n,i)(p,q);

		result /= rho;
		
  		return result;
  	}

	virtual arma::mat33 B(double S, arma::mat33 F) const
	{
		arma::mat33 G(FingerG(F));
		double G11=G(0,0), G12=G(0,1), G13=G(0,2), G22=G(1,1), G23=G(1,2), G33=G(2,2);		

		double rho = rho_0 * sqrt(det(G));

		arma::mat33 result;
		result.fill(-99999);
#               include "sympy/romenskii_dsigma_dS.inc"

		result /= rho;

		return result;
	}

	double S_to_potT(double S)
	{
		return exp(S/cv);
	}

	double potT_to_S(double potT)
	{
		return cv*log(potT);
	}

	Romenskii(double rho_0, double K0, double B0, double alpha, double beta,
		  double gamma, double cv, double T0)
		: rho_0(rho_0), K0(K0), B0(B0), alpha(alpha), beta(beta),
		  gamma(gamma), cv(cv), T0(T0)
	{}
};

}

#endif
