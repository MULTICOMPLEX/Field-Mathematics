// additional functions

template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
class multicomplex;

// Gamma function, Spouge's approximation

const int A = 35;

template <typename elem>
constexpr std::vector<elem> init_spouge() {

	std::vector<elem> Spouge_coeff(A);
	
	elem k1_factrl = 1; /* (k - 1)!*(-1)^k with 0!==1*/

	Spouge_coeff[0] = root_two_pi;

	for (int k = 1; k < A; k++)
	{
		Spouge_coeff[k] = exp(elem(A) - k) * pow(elem(A) - k, k - 0.5) / k1_factrl;
		k1_factrl *= -k;
	}
	return Spouge_coeff;
}


template <typename elem, int order>
multicomplex<elem, order> gamma
(
	const multicomplex<elem, order>& z
)
{
	auto Spouge_coeff = init_spouge<elem>();
	
	elem a = elem(A);

	multicomplex<elem, order> accm{ elem(Spouge_coeff[0]) };

	for (int64_t k = 1; k < A; k++)
	{
		accm = accm + Spouge_coeff[k] / (z + k);
	}

	accm = accm * exp(-(z + a)) * pow(z + a, z + elem(0.5)); /* Gamma(z+1) */

	return accm / z;
}


//https://github.com/IstvanMezo/LambertW-function

template <typename elem, int order>
multicomplex<elem, order> zexpz(const multicomplex<elem, order>& z)
{
	return z * exp(z);
}

//The derivative of z * exp(z) = exp(z) + z * exp(z)
template <typename elem, int order>
multicomplex<elem, order> zexpz_d(const multicomplex<elem, order>& z)
{
	return exp(z) + z * exp(z);
}

//The second derivative of z * exp(z) = 2. * exp(z) + z * exp(z)
template <typename elem, int order>
multicomplex<elem, order> zexpz_dd(const multicomplex<elem, order>& z)
{
	return 2. * exp(z) + z * exp(z);
}

//Determine the initial point for the root finding
template <typename elem, int order>
multicomplex<elem, order> InitPoint(int k, multicomplex<elem, order> z)
{

	const elem e{ 2.71828182845904523536 };
	multicomplex<elem, order> I{ 0, 1 };
	multicomplex<elem, order> two_pi_k_I{ 0., 2. * pi * k };
	multicomplex<elem, order> ip{ log(z) + two_pi_k_I - log(log(z) + two_pi_k_I) };// initial point coming from the general asymptotic approximation
	multicomplex<elem, order> p{ sqrt(2. * (e * z + 1.)) };// used when we are close to the branch cut around zero and when k=0,-1

	if (abs(z - (-exp(-1.))) <= 1.) //we are close to the branch cut, the initial point must be chosen carefully
	{
		if (k == 0) ip = -1. + p - 1. / 3. * pow(p, 2) + 11. / 72. * pow(p, 3);
		if (k == 1 && z.imag < 0.) ip = -1. - p - 1. / 3. * pow(p, 2) - 11. / 72. * pow(p, 3);
		if (k == -1 && z.imag > 0.) ip = -1. - p - 1. / 3. * pow(p, 2) - 11. / 72. * pow(p, 3);
	}

	if (k == 0 && abs(z - .5) <= .5) ip = (0.35173371 * (0.1237166 + 7.061302897 * z)) / (2. + 0.827184 * (1. + 2. * z));
	// (1,1) Pade approximant for W(0,a)

	if (k == -1 && abs(z - .5) <= .5) ip = -(((2.2591588985 +
		4.22096 * I) * ((-14.073271 - 33.767687754 * I) * z - (12.7127 -
			19.071643 * I) * (1. + 2. * z))) / (2. - (17.23103 - 10.629721 * I) * (1. + 2. * z)));// (1,1) Pade approximant for W(-1,a)

	return ip;
}

template <typename elem, int order>
multicomplex<elem, order> LambertW(int k = 0, multicomplex<elem, order> z = 0)
{
	//For some particular z and k W(z,k) has simple value:
	if (z == 0.) return (k == 0) ? 0. : -INFINITY;
	if (z == -exp(-1.) && (k == 0 || k == -1)) return -1.;
	if (z == exp(1.) && k == 0) return 1.;

	//Halley method begins
	multicomplex<elem, order> w{ InitPoint(k,z) }, wprev{ InitPoint(k,z) }; // intermediate values in the Halley method
	const unsigned int maxiter = 30; // max number of iterations. This eliminates improbable infinite loops
	unsigned int iter = 0; // iteration counter
	elem prec = 1.E-30; // difference threshold between the last two iteration results (or the iter number of iterations is taken)

	do
	{
		wprev = w;
		w -= 2. * ((zexpz(w) - z) * zexpz_d(w)) /
			(2. * pow(zexpz_d(w), 2) - (zexpz(w) - z) * zexpz_dd(w));
		iter++;
	} while ((abs(w - wprev) > prec) && iter < maxiter);
	return w;
}


















