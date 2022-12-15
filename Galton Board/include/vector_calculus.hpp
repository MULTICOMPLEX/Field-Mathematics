#include "matrix.hpp"

typedef std::chrono::high_resolution_clock Clock;

//Vector Analysis
//3D
namespace VX
{
	template<typename FX, typename FY, typename FZ, typename elem, int order>
	std::vector<multicomplex<elem, order>> gradient
	(
		FX fx,
		const multicomplex<elem, order>& x,
		FY fy,
		const multicomplex<elem, order>& y,
		FZ fz,
		const multicomplex<elem, order>& z
	)//∇f 
	{
		std::vector<multicomplex<elem, order>> v(3);
		multicomplex<elem, order + 1> dx, dy, dz;

		sh(dx, x);  v[0] = mcdv.dv<order>(fx(dx, y, z));
		sh(dy, y);  v[1] = mcdv.dv<order>(fy(x, dy, z));
		sh(dz, z);  v[2] = mcdv.dv<order>(fz(x, y, dz));

		return v;
	}

	//2D
	template<typename FX, typename FY, typename elem, int order>
	std::vector<multicomplex<elem, order>> gradient
	(
		FX fx,
		const multicomplex<elem, order>& x,
		FY fy,
		const multicomplex<elem, order>& y
	)//∇f 
	{
		std::vector<multicomplex<elem, order>> v(2);
		multicomplex<elem, order + 1> dx, dy;
		mcdv mcdv;

		mcdv.sh<order>(dx, x); v[0] = mcdv.dv<order>(fx(dx, y));//x^2 + xy +y^2
		mcdv.sh<order>(dy, y); v[1] = mcdv.dv<order>(fy(x, dy));

		return v;
	}

	template<typename FX, typename FY, typename FZ>
	std::vector<MX0> laplacian
	(
		FX fx,
		const MX0& x,
		FY fy,
		const MX0& y,
		FZ fz,
		const MX0& z
	)//Δf = ∇^2f = ∇·∇f 
	{
		std::vector<MX0> v(3);
		MX2 dx, dy, dz;

		sh(dx, x); v[0] = dv(fx(dx, y, z));
		sh(dy, y); v[1] = dv(fy(x, dy, z));
		sh(dz, z); v[2] = dv(fz(x, y, dz));

		return v;
	}

	template<typename F1, typename F2>
	std::vector<MX0> Laplacian
	(
		F1 f1,
		const MX0& x,
		F2 f2,
		const MX0& y
	)//Δf = ∇^2f = ∇·∇f 
	{
		std::vector<MX0> v(2);
		MX2 dx2, dy2;

		sh(dx2, x); v[0] = dv(f1(dx2, y)) + dv(f1(x, dy2));
		sh(dy2, y); v[1] = dv(f2(dx2, y)) + dv(f2(x, dy2));

		return v;
	}

	template<typename FX, typename FY>
	MX0 divergence
	(
		FX fx,
		const MX0& x,
		FY fy,
		const MX0& y
	)//∇.f 
	{
		std::vector<MX0> v(2);
		MX1 dx, dy;

		sh(dx, x); v[0] = dv(fx(dx, y));
		sh(dy, y); v[1] = dv(fy(x, dy));

		MX0 i = v[0] + v[1];
		return i;
	}


	template <typename T>
	T determinant
	(
		const std::vector<std::vector<T>>& matrix,
		int n
	)
	{
		T det{};
		std::vector<std::vector<T>> submatrix(matrix.front().size(), std::vector<T>(matrix.front().size()));
		if (n == 2)
			return ((matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1]));
		else {
			for (int x = 0; x < n; x++) {
				int subi = 0;
				for (int i = 1; i < n; i++) {
					int subj = 0;
					for (int j = 0; j < n; j++) {
						if (j == x)
							continue;
						submatrix[subi][subj] = matrix[i][j];
						subj++;
					}
					subi++;
				}
				det = det + (std::pow(-1, x)) * matrix[0][x] * determinant(submatrix, n - 1);
			}
		}
		return det;
	}

	template <typename T>
	T tr
	(
		const std::vector<std::vector<T>>& matrix
	)
	{
		T t;
		int n = int(matrix.front().size());

		for (int x = 0; x < n; x++) {
			for (int i = 0; i < n; i++) {
				if (i == x)t += matrix[x][i];
			}
		}
		return t;
	}

	template <typename T>
	std::vector<std::vector<T>> multiply_matrix
	(
		const std::vector<std::vector<T>>& mat1,
		const std::vector<std::vector<T>>& mat2
	)
	{
		std::vector<std::vector<T>> res(mat1.front().size(), std::vector<T>(mat1.front().size()));
		int N = int(mat1.front().size());

		int i, j, k;
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				res[i][j] = 0;
				for (k = 0; k < N; k++)
					res[i][j] += mat1[i][k] *
					mat2[k][j];
			}
		}
		return res;
	}

	template <typename T>
	std::vector<std::vector<T>> substract_matrix
	(
		const std::vector<std::vector<T>>& mat1,
		const std::vector<std::vector<T>>& mat2
	)
	{
		std::vector<std::vector<T>> res(mat1.front().size(), std::vector<T>(mat1.front().size()));
		int n = int(mat1.front().size());

		for (int x = 0; x < n; x++) {
			for (int i = 0; i < n; i++) {
				res[x][i] = mat1[x][i] - mat2[x][i];
			}
		}
		return res;
	}

	template<typename T>
	void print_matrix
	(
		const T& v
	)
	{
		//std::cout << std::showpos;
		for (auto& y : v) {
			for (auto& x : y)
				std::cout << x;
			std::cout << std::endl;
		}
	}

	template<typename F1, typename F2>
	MX0 Jacobian2x2
	(
		F1 f1,
		const MX0& x,
		F2 f2,
		const MX0& y
	)
	{
		MX1 dx, dy;

		sh(dx, x);
		sh(dy, y);

		std::vector<std::vector<MX0>> matrix(2, std::vector<MX0>(2));

		matrix[0][0] = dv(f1(dx, y));
		matrix[0][1] = dv(f1(x, dy));

		matrix[1][0] = dv(f2(dx, y));
		matrix[1][1] = dv(f2(x, dy));

		auto d = determinant(matrix, 2);

		print_matrix(matrix);
		std::cout << "determinant\n" << d << std::endl;

		return d;
	}

	template<typename F1, typename F2, typename F3>
	MX0 Jacobian3x3
	(
		F1 f1,
		const MX0& x,
		F2 f2,
		const MX0& y,
		F3 f3,
		const MX0& z
	)
	{
		std::vector<std::vector<MX0>> matrix(3, std::vector<MX0>(3));

		MX1 dx, dy, dz;

		sh(dx, x);
		sh(dy, y);
		sh(dz, z);

		matrix[0][0] = dv(f1(dx, y, z));
		matrix[0][1] = dv(f1(x, dy, z));
		matrix[0][2] = dv(f1(x, y, dz));

		matrix[1][0] = dv(f2(dx, y, z));
		matrix[1][1] = dv(f2(x, dy, z));
		matrix[1][2] = dv(f2(x, y, dz));

		matrix[2][0] = dv(f3(dx, y, z));
		matrix[2][1] = dv(f3(x, dy, z));
		matrix[2][2] = dv(f3(x, y, dz));

		auto d = determinant(matrix, 3);

		print_matrix(matrix);
		std::cout << "determinant\n" << d << std::endl;

		return d;
	}

	template<typename FX, typename FY, typename FZ>
	std::vector<MX0> curl
	(
		FX fx,
		const MX0& x,
		FY fy,
		const MX0& y,
		FZ fz,
		const MX0& z
	)//∇xf 
	{
		std::vector<MX0> v(3);
		MX0 a, b;
		MX1 dx, dy, dz;

		sh(dy, y); a = dv(fz(x, dy, z));
		sh(dz, z); b = dv(fy(x, y, dz));
		v[0] = a - b;

		sh(dz, z); a = dv(fx(x, y, dz));
		sh(dx, x); b = dv(fz(dx, y, z));
		v[1] = a - b;

		sh(dx, x); a = dv(fy(dx, y, z));
		sh(dy, y); b = dv(fx(x, dy, z));
		v[2] = a - b;

		return v;
	}

	template<typename F>
	MX0 Hessian2x2
	(
		F f,
		const MX0& x,
		const MX0& y
	)
	{
		MX1 dx1, dy1;
		MX2 dx2, dy2;

		sh(dx1, x);
		sh(dy1, y);

		sh(dx2, x);
		sh(dy2, y);

		Matrix<MX0> matrix(2,2), matrix2(2, 2), matrix3(2, 2);
		//std::vector<std::vector<MX0>> matrix(2, std::vector<MX0>(2));

		auto pd = [&](const auto& x, const auto& y) { return dv(f(x, y)); };//lambdify

		matrix[0][0] = dv(f(dx2, y));

		matrix[0][1] = pd(dx2, dy1) - pd(dx2, y);
		matrix[1][0] = matrix[0][1];//= symmetric

		matrix[1][1] = dv(f(x, dy2));

		//auto d = determinant(matrix, 2);
		auto determinant = det(matrix, 2);

		//print_matrix(matrix);
		std::cout << "determinant " << determinant << std::endl;
		
		std::cout << matrix << std::endl; 

		matrix.inverse(matrix2, 2, determinant);

		std::cout << matrix2 << std::endl;

		//std::cout << std::endl << "determinant\n" << d << std::endl;

		return determinant;
	}

	template<typename F, typename elem, int order>
	Matrix<multicomplex<elem, order>> inverse_Hessian2x2
	(
		F f,
		const multicomplex<elem, order>& x,
		const multicomplex<elem, order>& y
	)
	{
		multicomplex<elem, order + 1> dx1, dy1;
		multicomplex<elem, order + 2> dx2, dy2;
		mcdv mcdv;
		const int dv = order;//multicomplex order

		mcdv.sh<dv>(dx1, x);
		mcdv.sh<dv>(dy1, y);

		mcdv.sh<dv>(dx2, x);
		mcdv.sh<dv>(dy2, y);

		Matrix<multicomplex<elem, order>> matrix(2, 2), matrix2(2, 2), matrix3(2, 2);
		//std::vector<std::vector<MX0>> matrix(2, std::vector<MX0>(2));

		auto pd = [&](const auto& x, const auto& y) { return mcdv.dv<order>(f(x, y)); };//lambdify

		matrix[0][0] = mcdv.dv<dv>(f(dx2, y));

		matrix[0][1] = pd(dx2, dy1) - pd(dx2, y);
		matrix[1][0] = matrix[0][1];//= symmetric

		matrix[1][1] = mcdv.dv<dv>(f(x, dy2));

		//auto d = determinant(matrix, 2);
		auto determinant = det(matrix, 2);

		//print_matrix(matrix);
		//std::cout << "determinant " << determinant << std::endl;

		//std::cout << matrix << std::endl;

		matrix.inverse(matrix2, 2, determinant);

		//std::cout << matrix2 << std::endl;

		//std::cout << std::endl << "determinant\n" << d << std::endl;

		return matrix2;
	}

	template<typename F, typename elem, int order>
	multicomplex<elem, order> Hessian3x3
	(
		F f,
		const multicomplex<elem, order>& x,
		const multicomplex<elem, order>& y,
		const multicomplex<elem, order>& z
	)
	{
		multicomplex<elem, order + 1> dx1, dy1, dz1;
		multicomplex<elem, order + 2> dx2, dy2, dz2;

		sh(dx1, x);
		sh(dy1, y);
		sh(dz1, z);

		sh(dx2, x);
		sh(dy2, y);
		sh(dz2, z);

		//util::StartCounter();
		auto t1 = Clock::now();

		std::vector<std::vector<multicomplex<elem, order>>> matrix(3, std::vector<multicomplex<elem, order>>(3));

		auto pd = [&](const auto& x, const auto& y, const auto& z) { return dv(f(x, y, z)); };//lambdify

		matrix[0][0] = pd(dx2, y, z);
		matrix[0][1] = pd(dx2, dy1, z) - pd(dx2, y, z);
		matrix[0][2] = pd(dx1, y, dz2) - pd(x, y, dz2);

		matrix[1][0] = matrix[0][1];
		matrix[1][1] = pd(x, dy2, z);
		matrix[1][2] = pd(x, dy2, dz1) - pd(x, dy2, z);

		matrix[2][0] = matrix[0][2];
		matrix[2][1] = matrix[1][2];
		matrix[2][2] = pd(x, y, dz2);

		//auto stop = util::GetCounter(); 
		auto t2 = Clock::now();

		auto d = determinant(matrix, 3);

		print_matrix(matrix);

		auto p = std::cout.precision();

		std::cout.setf(std::ios::fixed, std::ios::floatfield);
		std::cout.precision(0);
		//std::cout << "\nDuration Hessian " << stop << " uS\n" << std::endl; 
		std::cout << "\nDuration Hessian " <<
			std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " uS\n" << std::endl;


		std::cout.precision(12);
		std::cout << "determinant\n" << d << std::endl;
		std::cout.precision(p);

		return d;
	}

	template <typename T>
	std::vector<T> normalize3D
	(
		std::vector<T> x
	)
	{
		T sqr = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
		sqr = sqrt(sqr);
		x[0] = x[0] * (1. / sqr);
		x[1] = x[1] * (1. / sqr);
		x[2] = x[2] * (1. / sqr);

		float k = std::sqrt(2) / 2.;
		std::vector<T> e{ {k,0.},{k,0.},{k,0.} };
		if (std::isnan(x[0].real))return e;
		if (std::isnan(x[1].real))return e;
		if (std::isnan(x[2].real))return e;

		return x;
	}

	template<typename T>
	std::vector<T> normalize2D
	(
		std::vector<T> x
	)
	{
		T sqr = x[0] * x[0] + x[1] * x[1];
		sqr = sqrt(sqr);

		x[0] = x[0] * (1. / sqr);
		x[1] = x[1] * (1. / sqr);

		auto k = std::sqrt(2) / 2.;
		std::vector<T> e{ {k,0.},{k,0.} };
		if (std::isnan(x[0].real))return e;
		if (std::isnan(x[1].real))return e;

		return x;
	}

	template<typename F>
	MX0 Directional_Derivative
	(
		F f,
		const MX1& point,
		const std::vector<MX0>& direction
	)
	{
		std::vector<MX0> v(2);
		MX1 dx, dy;

		sh(dx, point.real); v[0] = dv(f(dx, point.imag));
		sh(dy, point.imag); v[1] = dv(f(point.real, dy));

		return std::inner_product(v.begin(), v.end(), normalize2D(direction).begin(), MX0(0, 0));
		//return dot_product(v, normalize2D(direction));
	}

	template <typename T>
	T logint
	(
		T n
	)
	{
		size_t lg = 0;
		while ((1ull << lg) < n)
			lg++;
		return lg;
	}

	template <typename T>
	T rev
	(
		T x,
		T lgn
	)
	{
		T res = 0;
		while (lgn--) {
			res = 2 * res + (x & 1);
			x /= 2;
		}
		return res;
	}

	template <typename T>
	std::vector<T> FFT
	(
		const std::vector<T>& v
	)
	{
		typedef T Complex;
		const Complex J(0, 1);

		size_t n = v.size();
		size_t lgn = logint(n);
		assert((n & (n - 1)) == 0);
		std::vector<T> perm(v.size());

		for (size_t i = 0; i < n; i++) {
			perm[i] = v[rev(i, lgn)];
		}

		for (size_t s = 1; s <= lgn; s++) {
			size_t m = (1ull << s);
			T wm = exp(-2 * pi * J / m);
			for (size_t k = 0; k < n; k += m) {
				T w = 1;
				for (size_t j = 0; j < m / 2; j++) {
					T t = w * perm[k + j + m / 2];
					T u = perm[size_t(k) + j];
					perm[size_t(k) + j] = u + t;
					perm[size_t(k) + j + m / 2] = u - t;
					w = w * wm;
				}
			}
		}
		return perm;
	}

	template <typename T>
	std::vector<T> general_iFFT
	(
		const std::vector<T>& v,
		std::vector<T>(*f)(const std::vector<T>&))
	{
		std::vector<T> cp = v;
		for (auto& el : cp)
			el = T(el.real, -el.imag);

		cp = (*f)(cp);

		for (auto& el : cp)
			el = T(el.real, -el.imag) / v.size();
		return cp;

	}

	template <typename T>
	std::vector<T> iFFT
	(
		const std::vector<T>& v)
	{
		return general_iFFT(v, &FFT);
	}

	template <typename T>
	std::vector<T> recursive_FFT
	(
		const std::vector<T>& v)
	{
		typedef T Complex;
		const Complex J(0, 1);
		
		int n = v.size();
		if (n == 1)
			return v;

		assert((n & (n - 1)) == 0);

		T wn = exp(-2 * pi * J / n);
		T w = 1;

		std::vector<T> a0(n / 2), a1(n / 2);
		for (int i = 0; i < n; i += 2)
			a0[i / 2] = v[i], a1[i / 2] = v[i + 1];

		std::vector<T> y0, y1;
		y0 = recursive_FFT(a0);
		y1 = recursive_FFT(a1);
		std::vector<T> y(n);
		for (int k = 0; k < n / 2; k++)
		{
			y[k] = y0[k] + w * y1[k];
			y[k + n / 2] = y0[k] - w * y1[k];
			w = w * wn;
		}
		return y;
	}

	template <typename T>
	std::vector<T> recursive_iFFT
	(
		std::vector<T>& v
	)
	{
		return general_iFFT(v, &recursive_FFT);
	}

	template <typename T>
	T eval
	(
		const std::vector<T>& pol, T& point
	)
	{
		T result = 0;
		int n = pol.size();
		for (int i = 0; i < n; i++)
		{
			result = result * point;
			result = result + pol[n - 1 - i];
		}
		return result;
	}

	template <typename T>
	std::vector<T> ola_FFT
	(
		const std::vector<T>& v
	)
	{
		typedef T Complex;
		const Complex J(0, 1);

		int n = v.size();
		T wn = exp(-2 * pi * J / n);
		T w = 1;
		std::vector<T> result(n);

		for (int i = 0; i < n; i++)
		{
			result[i] = eval(v, w);
			w = w * wn;
		}
		return result;
	}

	template <typename T>
	std::vector<T> ola_iFFT
	(
		const std::vector<T>& v
	)
	{
		return general_iFFT(v, &ola_FFT);
	}

	template <typename T>
	void driver_fft()
	{
		/*
		std::vector<MX0> orig{
		{1,0},
		{1,0},
		{1,0},
		{1,0},
		{0,0},
		{0,0},
		{0,0},
		{0,0.9998},
		};
			*/
		std::vector<MX1> orig {
		 {{1,0},{0,0}},
		 {{1,0},{0,0}},
		 {{1,0},{0,0}},
		 {{1,0},{0,0}},
		 {{0,0},{0,0}},
		 {{0,0},{0,0}},
		 {{0,0},{0,0}},
		 {{0,0},{22,0.876}},
			 };

		std::cout << "orig" << std::endl;
		for (auto& i : orig)
			std::cout << i << std::endl;
		std::cout << std::endl;

			 // forward fft
		auto dft = VX::FFT(orig);

		std::cout << "fft" << std::endl;
		for (int i = 0; i < 8; ++i)
		{
			std::cout << dft[i] << std::endl;
		}

		// inverse fft
		auto idft = VX::iFFT(dft);

		std::cout << std::endl << "ifft" << std::endl;
		for (int i = 0; i < 8; ++i)
		{
			std::cout << idft[i] << std::endl;
		}

	}

	template <typename T>
	void vc_eval
	(
	)
	{
		auto fx1 = [](const auto& x, const auto& y) { return x * x + x * sin(y); };//∇f, grad x^2 + xy +y^2
		auto fy1 = [](const auto& x, const auto& y) { return y * y + sin(y) * x; };//The Del, or ‘Nabla’ operator 
		//it may denote the gradient of a scalar field, the divergence of a vector field, or the curl of a vector field.
		auto v2 = gradient(fx1, /*x*/MX0{ .7,0 }, fy1, /*y*/MX0{ 1,0 });
		std::cout << "\nGradient ∇ F(x*x+x*sin(y)), F(y*y+sin(y)*x), x=.7+0i, y=1.0+0i" << std::endl;
		for (auto& i : v2)
			std::cout << i << std::endl;//∇f 

		auto fx2 = [](const auto& x, const auto& y) { return x * x + x * y; };
		auto fy2 = [](const auto& x, const auto& y) { return y * y + y * x * x; };//y*y+y*x^2
		std::cout << "\nLaplacian ∇·∇ =∇=  F(x*x+x*y), F(y*y+y*x^2), x=1+0i, y=1+0i" << std::endl;
		auto v1 = Laplacian(fx2,/*x*/{ 1,0 }, fy2, /*y*/{ 1,0 });
		for (auto& i : v1)
			std::cout << i << std::endl;//∇·∇ =∇= f

		auto fx3 = [](const auto& x, const auto& y) { return x * x - y * y; };//∇.f, del x^2-y^2, 2xy
		auto fy3 = [](const auto& x, const auto& y) { return 2 * sin(x) * y; };
		std::cout << "\nDivergence ∇·  F(x*x-y*y), F(2*sin(x)*y), x=.7+0i, y=1.0+0i" << std::endl;
		std::cout << divergence(fx3, /*x*/{ .7,0 }, fy3, /*y*/{ 1.,0 }) << std::endl;//div{x^2 - y^2, 2 x y} = 4 x

		//∇xf, curl [-y/(x^2+y^2)z, -x/(x^2+y^2)z, z]
		auto fx4 = [](const auto& x, const auto& y, const auto& z) { return -y / (x * x + y * y) * z; };
		auto fy4 = [](const auto& x, const auto& y, const auto& z) { return -x / (x * x + y * y) * z; };
		auto fz = [](const auto& x, const auto& y, const auto& z) { return z; };

		auto vf3 = curl(fx4, /*x*/{ .7,0 }, fy4, /*y*/{ 1,0 }, fz, /*z*/{ 1,0 });

		std::cout << "\nCurl ∇×  F(-y/(x*x+y*y)*z), F(-x/(x*x+y*y)*z), x=.7+0i, y=1.0+0i, z=1.0+0i" << std::endl;
		for (auto& i : vf3)
			std::cout << i << std::endl;

		{
			auto fx = [](const auto& x, const auto& y) { return x * x * y; };//∇.f, del x^2-y^2, 2xy
			auto fy = [](const auto& x, const auto& y) { return 5 * x + sin(y); };

			std::cout << "\nJacobian2x2  F1(x*x*y), F2(5*x+sin(y)), x=.7+0i, y=1.0+0i\n";
			Jacobian2x2(fx, /*x*/{ .7,0 }, fy, /*y*/{ 1,0 });
		}
		{
			//Jacobian of (x x y z z, 5 x+sin(y) z, 5 x+cos(z)+y)
			//(x z ((5 x - 2 y) z sin(y) - 2 y z cos(y) (5 x + z sin(z)) + 5 x (2 y + z sin(z)))), x = (.7+0i), y = (1.0+0i), z = (1.0+0.8i)
			auto fx = [](const auto& x, const auto& y, const auto& z) { return x * x * y * z * z; };
			auto fy = [](const auto& x, const auto& y, const auto& z) { return 5 * x + sin(y) * z; };
			auto f3 = [](const auto& x, const auto& y, const auto& z) { return 5 * x + cos(z) + y; };
			std::cout << "\nJacobian3x3  F1(x*x*y), F2(5*x+sin(y)), F3(5*x+cos(z)) x=.7+0i, y=1.0+0i, z=1.0+0.8i\n";
			Jacobian3x3(fx, /*x*/{ .7,0 }, fy, /*y*/{ 1,0 }, f3, /*z*/{ 1,0.8 });
		}

		auto f1 = [](const auto& x, const auto& y) { return x * x + x * x + 3 * y * y * y - x; };
		std::cout << "\nHessian2x2  F(x*x+x*x+3*y*y*y-x), x=.7+0i, y=.9+0i\n";
		Hessian2x2(f1, /*x*/{ .7,0 }, /*y*/{ .9,0 });

		auto f2 = [](const auto& x, const auto& y, const auto& z) { return x * z / x + sqrt(x * z * y * x + y * z) * pow(y, 6) * x * z; };
		std::cout << "\nHessian3x3  F(x*z*x+sqrt(x*z*y*x+y*z)*pow(y,6)*x*z), x=.7+0i, y=.9+0i, z=.2+0i" << std::endl;
		Hessian3x3(f2, /*x*/MX0{ .7,0 }, /*y*/MX0{ .9,0 }, /*z*/MX0{ .2,0 });

		//Directional Derivative
		auto f3 = [](const auto& x, const auto& y) { return 3 * (x * x) * y; };
		std::cout << "\nDirectional Derivative  F(3(x^2)y), x=1+0i, y=2+0i, U1= 3/5+0i, 4/5+0i\n";

		std::cout.setf(std::ios::fixed, std::ios::floatfield);
		auto p = std::cout.precision(12);

		std::vector<MX0> direction{ {1,0},{2,0} };
		auto ddv = Directional_Derivative(f3, /*point x,y*/{ {3. / 5.,0},{4. / 5.,0} }, /*vector*/ direction);
		std::cout << ddv << std::endl << std::endl;
		std::cout.precision(p);

	}

	template <typename F1, typename elem, int order>
	std::vector<multicomplex<elem, order>> Critical_point_2
	(
		F1 f,
		const multicomplex<elem, order>& x,
		const multicomplex<elem, order>& y,
		int n
	)
	{
		int tel = 0;

		std::vector<multicomplex<elem, order>> h(2), v = { x,y };

		while (tel <= n)
		{
			h = gradient(f, v[0], f, v[1]) * inverse_Hessian2x2(f, v[0], v[1]);

			// x(i+1) = x(i) - f'(x) / f''(x)   
			v -= h;

			printf("%02d ", tel);
			std::cout << v << std::endl;
			tel++;
		}

		return v;
	}

}
