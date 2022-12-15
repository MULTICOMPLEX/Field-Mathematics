
#include <vector>
#include <iostream>
#include <array>

template <typename elem, int order>
class multicomplex;

template <class T>
class Matrix
{
public:

	enum { Rows = 32, Cols = 32 };
	T matrix[Rows][Cols] = {};
	//T** matrix;
	size_t height = 2;
	size_t width = 2;

	Matrix(size_t height, size_t width);
	Matrix();
	Matrix(std::initializer_list<std::initializer_list<T>> listlist);

	virtual ~Matrix() = default;

	size_t getHeight() const;
	size_t getWidth() const;

	Matrix identity();

	Matrix add(Matrix& m) const;
	Matrix subtract(Matrix& m) const;

	Matrix dot(const Matrix& m) const;
	Matrix transpose() const;
	Matrix multiply(const T& value) const;
	Matrix divide(const T& value) const;

	Matrix divide(Matrix& m) const;

	Matrix applyFunction(T(*function)(T)) const;
	Matrix subMat(size_t startH, size_t startW, size_t h, size_t w) const;

	void fill(const T& value);
	void put(size_t h, size_t w, const T& value);
	T get(size_t h, size_t w) const;

	void Print
	(
		std::ostream& flux,
		Matrix<T> m
	) const;

	bool operator==(const Matrix& m);
	bool operator!=(const Matrix& m);
	Matrix operator+=(const Matrix& m);
	Matrix operator-=(const Matrix& m);
	Matrix operator*=(const Matrix& m);
	Matrix operator*=(const T& m);

	inline T(&operator [](size_t i))[Cols]
	{
		return matrix[i];
	}

	Matrix operator/=(const T& m);


	T& operator()(int y, int x);

	Matrix operator* (Matrix& b);

	void transpose(Matrix& c, Matrix& d, int n, T determinant);
	void cofactor(Matrix& a, Matrix& d, int n, T determinant);
	void inverse(Matrix& d, int n, T determinant);

	void minor(Matrix& b, Matrix& a, int i, int n);

	T tr();
};

//	calculate minor of matrix OR build new matrix : k-had = minor

template <class T>
inline void Matrix<T>::minor
(
	Matrix& b,
	Matrix& a,
	int i,
	int n
)
{
	int j, l, h = 0, k = 0;
	for (l = 1; l < n; l++)
		for (j = 0; j < n; j++) {
			if (j == i)
				continue;
			b[h][k] = a[l][j];
			k++;
			if (k == (n - 1)) {
				h++;
				k = 0;
			}
		}
}// end function

//---------------------------------------------------

//	calculate determinant of matrix
template <class T>
inline T det
(
	Matrix<T>& a,
	int n
)
{
	int i;
	auto s = a.height;
	Matrix<T> b(s, s);
	T sum = 0;
	if (n == 1)
		return a[0][0];
	else if (n == 2)
		return (a[0][0] * a[1][1] - (a[0][1] * a[1][0]));
	else
		for (i = 0; i < n; i++) {
			a.minor(b, a, i, n);	// read function
			sum = (T)(sum + a[0][i] * std::pow(-1, i) * det(b, (n - 1)));	// read function	// sum = determinte matrix
		}
	return sum;
}// end function

//---------------------------------------------------

//	calculate transpose of matrix
template <class T>
inline void Matrix<T>::transpose
(
	Matrix& c,
	Matrix& d,
	int n,
	T determinant
)
{
	int i, j;
	auto s = c.height;

	Matrix b(s, s);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			b[i][j] = c[j][i];
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			d[i][j] = b[i][j] / determinant;	// array d[][] = inverse matrix
}// end function

//---------------------------------------------------

//	calculate cofactor of matrix
template <class T>
inline void Matrix<T>::cofactor
(
	Matrix& a,
	Matrix& d,
	int n,
	T determinant
)
{
	auto t = a.height;
	Matrix b(t, t), c(t, t);
	int l, h, m, k, i, j;
	for (h = 0; h < n; h++)
		for (l = 0; l < n; l++) {
			m = 0;
			k = 0;
			for (i = 0; i < n; i++)
				for (j = 0; j < n; j++)
					if (i != h && j != l) {
						b[m][k] = a[i][j];
						if (k < (n - 2))
							k++;
						else {
							k = 0;
							m++;
						}
					}
			c[h][l] = (T)std::pow(-1, (h + l)) * det(b, (n - 1));	// c = cofactor Matrix
		}
	transpose(c, d, n, determinant);	// read function
}// end function

//---------------------------------------------------

//	calculate inverse of matrix
template <class T>
inline void Matrix<T>::inverse
(
	Matrix& d,
	int n,
	T determinant
)
{
	if (determinant == 0)
	{
		std::cout << "\nInverse of Entered Matrix is not possible\n";
		exit(1);
	}
	else if (n == 1)
		d[0][0] = 1;
	else
		cofactor(*this, d, n, determinant); // read function
}// end function

//---------------------------------------------------

template <class T>
inline Matrix<T>::Matrix
(
	size_t h,
	size_t w
) : height(h), width(w)
{
	if (height > 32) height = 32;
	if (width > 32) width = 32;
}

//---------------------------------------------------

template <class T>
inline Matrix<T>::Matrix
(
)
{
	height = 2;
	width = 2;
}
//---------------------------------------------------

template <typename T>
inline Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> listlist) :
	height(listlist.begin()->size()), width(listlist.size()) {

	auto rows = height;
	auto cols = width;

	auto m = new T * [rows];

	for (size_t i = 0; i < rows; i++) {
		m[i] = new T[cols];
		for (size_t j = 0; j < cols; j++) {
			m[i][j] = ((listlist.begin() + i)->begin())[j];
		}
	}

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			matrix[i][j] = m[i][j];
		}
	}
}
//---------------------------------------------------

template <class T>
inline size_t Matrix<T>::getHeight
(
) const
{
	return height;
}

//---------------------------------------------------

template <class T>
inline size_t Matrix<T>::getWidth
(
) const
{
	return width;
}

//---------------------------------------------------

template <class T>
inline void Matrix<T>::fill
(
	const T& value
)
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			matrix[i][j] = value;
		}
	}
}

//---------------------------------------------------

template <class T>
inline T Matrix<T>::tr
(
)
{
	T sum = 0;
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			if (i == j)sum += matrix[i][j];
		}
	}
	return sum;
}

//---------------------------------------------------

template <class T>
inline void Matrix<T>::put
(
	size_t h,
	size_t w,
	const T& value
)
{
	if (!(h >= 0 && h < height && w >= 0 && w < width))
		throw std::invalid_argument("Index out of bounds.");

	matrix[h][w] = value;
}

//---------------------------------------------------

template <class T>
inline T Matrix<T>::get
(
	size_t h,
	size_t w
) const
{
	if (!(h >= 0 && h < height && w >= 0 && w < width))
		throw std::invalid_argument("Index out of bounds.");

	return matrix[h][w];
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::multiply
(
	const T& value
) const
{
	Matrix result(*this.matrix);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result[i][j] *= value;
		}
	}

	return result;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::divide
(
	const T& value
) const
{
	Matrix result = *this;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result[i][j] /= value;
		}
	}

	return result;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::divide
(
	Matrix<T>& m
) const
{
	Matrix<T> result(height, width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result[i][j] = matrix[i][j] / m[i][j];
		}
	}

	return result;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::add
(
	Matrix<T>& m
) const
{
	if (!(height == m.height && width == m.width))
		throw std::invalid_argument("Matrix dimension must be the same.");

	Matrix<T> result(height, width);
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			result[i][j] = matrix[i][j] + m[i][j];
		}
	}
	return result;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::subtract
(
	Matrix<T>& m
) const
{
	if (!(height == m.height && width == m.width))
		throw std::invalid_argument("Matrix dimension must be the same.");

	Matrix<T> result(height, width);
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			result[i][j] = matrix[i][j] - m[i][j];
		}
	}
	return result;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::dot
(
	const Matrix& m
) const
{
	if (!(width == m.height))
		throw std::invalid_argument("Dot product not compatible.");

	T w = 0;
	int mwidth = m.width;

	Matrix result(height, mwidth);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < mwidth; j++) {
			for (int h = 0; h < width; h++) {
				w += matrix[i][h] * m[h][j];
			}
			result[i][j] = w;
			w = 0;
		}
	}
	return result;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::transpose
(
) const
{
	Matrix result(width, height);

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			result[i][j] = matrix[j][i];
		}
	}
	return result;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::applyFunction
(
	T(*function)(T)
) const
{
	Matrix result(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result[i][j] = (*function)(matrix[i][j]);
		}
	}
	return result;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::identity
(
)
{
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width; j++)
		{
			if (i == j)matrix[i][j] = T(1);
		}
	}
	return *this;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::subMat
(
	size_t startH,
	size_t startW,
	size_t h,
	size_t w
) const
{
	if (!(startH >= 0 && startH + h <= height && startW >= 0 && startW + w <= width))
		throw std::invalid_argument("Index out of bounds");

	Matrix result(h, w);
	for (int i = startH; i < startH + h; i++)
	{
		for (int j = startW; j < startW + w; j++)
		{
			result[i - startH][j - startW] = matrix[i][j];
		}
	}
	return result;
}

//---------------------------------------------------

template <class T>
inline bool Matrix<T>::operator==
(
	const Matrix& m
	)
{
	if (height == m.height && width == m.width)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (matrix[i][j] != m[i][j])
				{
					return false;
				}
			}
		}
		return true;
	}
	return false;
}

//---------------------------------------------------

template <class T>
inline bool Matrix<T>::operator!=
(
	const Matrix& m
	)
{
	return !operator==(m);
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::operator+=
(
	const Matrix& m
	)
{
	*this = add(m);
	return *this;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::operator-=
(
	const Matrix& m
	)
{
	*this = subtract(m);
	return *this;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::operator*=
(
	const Matrix& m
	)
{
	*this = multiply(m);
	return *this;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::operator*=
(
	const T& m
	)
{
	*this = multiply(m);
	return *this;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::operator/=
(
	const T& m
	)
{
	*this = divide(m);
	return *this;
}

//---------------------------------------------------

template <class T>
inline T& Matrix<T>::operator()
(
	int y,
	int x
	)
{
	if (!(y >= 0 && y < height && x >= 0 && x < width))
		throw std::invalid_argument("Index out of bounds.");
	return matrix[y][x];
}

//---------------------------------------------------

template <class T>
Matrix<T> operator/
(
	const Matrix<T>& a,
	Matrix<T> b
	)
{
	return a.divide(b);
}

//---------------------------------------------------

template <class T>
inline Matrix<T> operator+
(
	const Matrix<T>& a,
	Matrix<T> b
	)
{
	return a.add(b);
}

//---------------------------------------------------

template <class T>
Matrix<T> operator-
(
	const Matrix<T>& a,
	Matrix<T> b
	)
{
	return a.subtract(b);
}

//---------------------------------------------------

template <class elem>
inline const Matrix<elem> operator -
(
	const elem& a,
	const Matrix<elem>& b
	)
{
	const Matrix<elem> n(a);
	return n - b;
}

//---------------------------------------------------

template <class elem>
inline const Matrix<elem> operator -
(
	const Matrix<elem>& a,
	const elem& b
	)
{
	const Matrix<elem> n(b);
	return a - n;
}

//---------------------------------------------------

template <class T>
inline Matrix<T> Matrix<T>::operator*
(
	Matrix<T>& b
	)
{
	size_t h = height;
	size_t w = width;

	Matrix<T> result(h, w);

	for (size_t i = 0; i < h; i++)
	{
		for (size_t j = 0; j < w; j++)
		{
			result[i][j] = 0;
			for (size_t k = 0; k < w; k++)
				result[i][j] += matrix[i][k] * b[k][j];
		}
	}

	return result;
}

//---------------------------------------------------

template <typename T, typename elem>
inline const Matrix<elem> operator*
(
	const T& a,
	const Matrix<elem>& b
	)
{
	auto h = b.height;
	auto w = b.width;

	Matrix<elem> result(b);

	for (size_t i = 0; i < h; i++)
	{
		for (size_t j = 0; j < w; j++)
		{
			result[i][j] *= elem(a);
		}
	}

	return result;
}

//---------------------------------------------------

template <typename elem, int order>
inline const Matrix<multicomplex<elem, order>> operator*
(
	const multicomplex<elem, order>& a,
	const Matrix<multicomplex<elem, order>>& b
	)
{
	auto h = b.height;
	auto w = b.width;

	Matrix<multicomplex<elem, order>> result(b);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			result[i][j] *= a;
		}
	}

	return result;
}


template <typename elem>
inline const Matrix<multicomplex<elem, 0>> operator*
(
	const multicomplex<elem, 0>& a,
	const Matrix<multicomplex<elem, 0>>& b
	)
{
	auto h = b.height;
	auto w = b.width;

	Matrix<multicomplex<elem, 0>> result(b);

	for (size_t i = 0; i < h; i++)
	{
		for (size_t j = 0; j < w; j++)
		{
			result[i][j] *= a;
		}
	}

	return result;
}

//---------------------------------------------------

template <typename T, typename elem>
inline const Matrix<elem> operator*
(
	const Matrix<elem>& b,
	const T& a
	)
{
	auto h = b.height;
	auto w = b.width;

	Matrix<elem> result(b);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			result[i][j] *= elem(a);
		}
	}

	return result;
}

//---------------------------------------------------

template <class elem>
inline const std::vector<elem> operator*
(
	const Matrix<elem>& a,
	const std::vector<elem>& b
	)
{
	std::size_t i, j;
	std::size_t m = a.width;
	std::size_t n = b.size();

	std::vector<elem> prod(m);

	for (i = 0; i < m; i++) {
		prod[i] = 0.;
		for (j = 0; j < n; j++)
			prod[i] += a[i][j] * b[j];
	}
	return prod;
}

//---------------------------------------------------

template <class T>
inline const std::vector<T> operator*
(
	const std::vector<T>& a,
	const Matrix<T>& b
	)
{
	std::size_t i, j;
	std::size_t m = b.width;
	std::size_t n = a.size();

	std::vector<T> prod(m);

	for (i = 0; i < m; i++) {
		prod[i] = 0.;
		for (j = 0; j < n; j++)
			prod[i] += b.matrix[i][j] * a[j];
	}
	return prod;
}

//---------------------------------------------------

template <typename T>
inline const std::vector<T> operator+
(
	const std::vector<T>& a,
	const std::vector<T>& b
	)
{
	std::vector<T> res = a;

	for (size_t i = 0; i < a.size(); i++)
		res[i] += b[i];

	return res;
}

template <typename T>
inline const std::vector<T> operator+
(
	const std::vector<T>& a,
	const T& b
	)
{
	std::vector<T> res;

	for (size_t i = 0; i < a.size(); i++)
		res[i] = a[i] + b;

	return res;
}

template <typename T>
inline const std::vector<T> operator+=
(
	std::vector<T>& a,
	const std::vector<T>& b
	)
{
	for (size_t i = 0; i < a.size(); i++)
		a[i] += b[i];
	return a;
}

template <typename T>
inline const std::vector<T> operator+=
(
	std::vector<T>& a,
	const T& b
	)
{
	for (size_t i = 0; i < a.size(); i++)
		a[i] += b;
	return a;
}

template <typename T>
inline const std::vector<T> operator+
(
	const std::vector<T>& a,
	const MX0& b
	)
{
	std::size_t i;

	std::vector<T> res = a;

	for (i = 0; i < a.size(); i++)
		res[i] += b;

	return res;
}

template <typename T>
inline const std::vector<T> operator+=
(
	std::vector<T>& a,
	const MX0& b
	)
{
	a = a + b;
	return a;
}

template <typename T>
inline const std::vector<T> operator-
(
	const std::vector<T>& a,
	const std::vector<T>& b
	)
{
	std::vector<T> res = a;

	for (size_t i = 0; i < b.size(); i++)
		res[i] -= b[i];

	return res;
}

template <typename T>
inline const std::vector<T> operator-=
(
	std::vector<T>& a,
	const T& b
	)
{
	for (size_t i = 0; i < a.size(); i++)
		a[i] -= b;
	return a;
}

template <typename T>
inline const std::vector<T> operator-=
(
	std::vector<T>& a,
	const std::vector<T>& b
	)
{
	a = a - b;
	return a;
}


template <typename T>
inline const std::vector<T> operator-
(
	const std::vector<T>& a,
	const MX0& b
	)
{
	std::size_t i;

	std::vector<T> res = a;

	for (i = 0; i < a.size(); i++)
		res[i] -= b;

	return res;
}

template <typename T>
inline const std::vector<T> operator-=
(
	std::vector<T>& a,
	const MX0& b
	)
{
	a = a - b;
	return a;
}

//---------------------------------------------------

template <typename T>
inline const std::vector<T> operator*
(
	const std::vector<T>& a,
	const std::vector<T>& b
	)
{
	std::size_t i;

	std::vector<T> res = a;

	for (i = 0; i < b.size(); i++)
		res[i] *= b[i];

	return res;
}

template <typename T>
inline const std::vector<T> operator*=
(
	std::vector<T>& a,
	const std::vector<T>& b
	)
{
	a = a * b;
	return a;
}


template <typename T>
inline const std::vector<T> operator*
(
	const std::vector<T>& a,
	const MX0& b
	)
{
	std::size_t i;

	std::vector<T> res = a;

	for (i = 0; i < a.size(); i++)
		res[i] *= b.real;

	return res;
}

template <typename T>
inline const std::vector<T> operator*
(
	const MX0& a,
	const std::vector<T>& b
	)
{
	std::size_t i;

	std::vector<T> res = b;

	for (i = 0; i < b.size(); i++)
		res[i] *= a.real;

	return res;
}

template <typename T>
inline const std::vector<T> operator*
(
	const std::vector<T>& a,
	const T& b
	)
{
	std::size_t i;

	std::vector<T> res = a;

	for (i = 0; i < a.size(); i++)
		res[i] *= b;

	return res;
}

template <typename T>
inline const std::vector<T> operator*
(
	const T& a,
	const std::vector<T>& b

	)
{
	std::vector<T> res = b;

	for (size_t i = 0; i < b.size(); i++)
		res[i] *= a;

	return res;
}

template <typename T>
inline const std::vector<T> operator*=
(
	std::vector<T>& a,
	const MX0& b
	)
{
	a = a * b.real;
	return a;
}

template <typename T>
inline const std::vector<MX0> operator*=
(
	std::vector<MX0>& a,
	const T& b
	)
{
	a = a * b;
	return a;
}


template <typename T>
inline const std::vector<T> operator*=
(
	std::vector<T>& a,
	const T& b
	)
{
	a = a * b;
	return a;
}


//---------------------------------------------------

template <typename T>
inline const std::vector<T> operator/
(
	const std::vector<T>& a,
	const std::vector<T>& b
	)
{
	std::size_t i;

	std::vector<T> res = a;

	for (i = 0; i < b.size(); i++)
		res[i] /= b[i];

	return res;
}

template <typename T>
inline const std::vector<T> operator/=
(
	std::vector<T>& a,
	const T& b
	)
{
	std::size_t i;

	for (i = 0; i < a.size(); i++)
		a[i] /= b;

	return a;
}

template <typename T>
inline const std::vector<T> operator/=
(
	std::vector<T>& a,
	const int& b
	)
{
	std::size_t i;

	for (i = 0; i < a.size(); i++)
		a[i] /= b;

	return a;
}

template <typename T>
inline const std::vector<T> operator/=
(
	std::vector<T>& a,
	const std::vector<T>& b
	)
{
	a = a / b;
	return a;
}

template <typename T>
inline const std::vector<T> operator/
(
	const T& a,
	const std::vector<T>& b

	)
{
	std::vector<T> res = b;

	for (size_t i = 0; i < b.size(); i++)
		res[i] /= a;

	return res;
}

template <typename T>
inline const std::vector<T> operator/
(
	const std::vector<T>& a,
	const T& b

	)
{
	std::vector<T> res = a;

	for (size_t i = 0; i < a.size(); i++)
		res[i] /= b;

	return res;
}

template <typename T>
inline const std::vector<T> operator/
(
	const std::vector<T>& a,
	const int& b

	)
{
	std::vector<T> res = a;

	for (size_t i = 0; i < a.size(); i++)
		res[i] /= T(b);

	return res;
}

//---------------------------------------------------

template <class T>
inline void Matrix<T>::Print
(
	std::ostream& flux,
	Matrix<T> m
) const
{
	for (size_t i = 0; i < m.height; i++)
	{
		for (size_t j = 0; j < m.width; j++)
		{
			flux << m[i][j] << " ";
		}
		flux << std::endl;
	}
}

//---------------------------------------------------

template <class elem>
inline std::ostream& operator <<
(
	std::ostream& flux,
	const std::vector<elem>& a
	)
{

	for (auto& i : a)
		flux << " " << i << std::endl;

	return flux;
}

template <typename elem, int order>
inline std::ostream& operator <<
(
	std::ostream& flux,
	const std::vector<multicomplex<elem, order>>& a
	)
{
	for (auto& i : a)
		flux << " " << i;

	return flux;
}
//---------------------------------------------------

template <class T>
inline std::ostream& operator <<
(
	std::ostream& flux,
	const Matrix<T>& m
	)
{
	m.Print(flux, m);
	return flux;
}

//---------------------------------------------------

template <class elem>
inline std::vector<elem> normalize
(
	const std::vector<elem>& arr
)
{
	std::vector<elem> output(arr.size());
	elem mod = 0.0;

	for (std::size_t i = 0; i < arr.size(); ++i) {
		mod += arr[i] * arr[i];
	}

	elem mag = std::sqrt(mod);

	if (mag == 0) {
		throw std::logic_error("The input vector is a zero vector");
	}

	for (std::size_t i = 0; i < arr.size(); ++i) {
		output[i] = arr[i] / mag;
	}

	return output;
}

//---------------------------------------------------

template<typename elem, int order, size_t num_in>
std::array<multicomplex<elem, order>, num_in> linspace(const elem start_in, const elem end_in)
{
	std::array<multicomplex<elem, order>, num_in> linspaced;

	elem start = start_in;
	elem end = end_in;
	elem num = num_in;

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced[0] = (start);
		return linspaced;
	}

	elem delta = (end - start) / (num - 1);

	for (size_t i = 0; i < num - 1; ++i)
	{
		linspaced[i] = (start + delta * i);
	}
	linspaced[num_in - 1] = end; // I want to ensure that start and end
														// are exactly the same as the input
	return linspaced;
}

//---------------------------------------------------

template<typename T, size_t num_in>
std::array<T, num_in> linspace(const T start_in, const T end_in)
{
	std::array<T, num_in> linspaced;

	T start = start_in;
	T end = end_in;
	T num = num_in;

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced[0] = (start);
		return linspaced;
	}

	T delta = (end - start) / (num - 1);

	for (size_t i = 0; i < num - 1; ++i)
	{
		linspaced[i] = (start + delta * i);
	}
	linspaced[num_in - 1] = end; // I want to ensure that start and end
														// are exactly the same as the input
	return linspaced;
}


//---------------------------------------------------

template <typename T, size_t N>
void print_array(std::array<T, N>& vec)
{
	for (auto& d : vec)
		std::cout << d << " ";
	std::cout << std::endl;
}


//---------------------------------------------------

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr)
{
	os.setf(std::ios::fixed, std::ios::floatfield);
	os.precision(10);

	int i = 0;
	os << "[";
	for (auto d = arr.begin(); d != arr.end(); ++d)
	{
		os << std::setw(14) << *d;
		if (d != arr.end() - 1) {
			os << " ";
			if (i++ == 4) { i = 0; os << std::endl << " "; }

		}
		else { os << "]"; }
	}

	return os;
}

template <typename scalar_type> class matrix {
public:
	matrix(size_t rows, size_t columns)
		: rows_(rows), columns_(columns), elements_(rows* columns) {}
	matrix(size_t rows, size_t columns,
		const std::initializer_list<std::initializer_list<scalar_type>>& values)
		: rows_(rows), columns_(columns), elements_(rows* columns) {
		assert(values.size() <= rows_);
		size_t i = 0;
		for (const auto& row : values) {
			assert(row.size() <= columns_);
			std::copy(begin(row), end(row), &elements_[i]);
			i += columns_;
		}
	}
	size_t rows() const { return rows_; }
	size_t columns() const { return columns_; }

	const scalar_type& operator()(size_t row, size_t column) const {
		assert(row < rows_);
		assert(column < columns_);
		return elements_[row * columns_ + column];
	}
	scalar_type& operator()(size_t row, size_t column) {
		assert(row < rows_);
		assert(column < columns_);
		return elements_[row * columns_ + column];
	}
private:
	size_t rows_;
	size_t columns_;
	std::vector<scalar_type> elements_;
};

// See https://en.wikipedia.org/wiki/Kronecker_product
template <typename scalar_type>
matrix<scalar_type> kronecker_product(const matrix<scalar_type>& a,
	const matrix<scalar_type>& b) {
	size_t arows = a.rows();
	size_t acolumns = a.columns();
	size_t brows = b.rows();
	size_t bcolumns = b.columns();
	matrix<scalar_type> c(arows * brows, acolumns * bcolumns);
	for (size_t i = 0; i < arows; ++i)
		for (size_t j = 0; j < acolumns; ++j)
			for (size_t k = 0; k < brows; ++k)
				for (size_t l = 0; l < bcolumns; ++l)
					c(i * brows + k, j * bcolumns + l) = a(i, j) * b(k, l);
	return c;
}

template <typename scalar_type>
void print(std::wostream& out, const matrix<scalar_type>& a) {
	const wchar_t* box_top_left = L"\x250c";
	const wchar_t* box_top_right = L"\x2510";
	const wchar_t* box_bottom_left = L"\x2514";
	const wchar_t* box_bottom_right = L"\x2518";
	const wchar_t* box_vertical = L"\x2502";
	const wchar_t nl = L'\n';
	const wchar_t space = L' ';
	const int width = 2;

	size_t rows = a.rows(), columns = a.columns();
	std::wstring spaces((width + 1) * columns, space);
	out << box_top_left << spaces << box_top_right << nl;
	for (size_t row = 0; row < rows; ++row) {
		out << box_vertical;
		for (size_t column = 0; column < columns; ++column)
			out << std::setw(width) << a(row, column) << space;
		out << box_vertical << nl;
	}
	out << box_bottom_left << spaces << box_bottom_right << nl;
}

void testk1() {
	std::wcout.imbue(std::locale(""));
	matrix<int> matrix1(2, 2, { {1,2}, {3,4} });
	matrix<int> matrix2(2, 2, { {0,5}, {6,7} });
	std::wcout << L"Test case 1:\n";
	print(std::wcout, kronecker_product(matrix1, matrix2));
}

void testk2() {
	std::wcout.imbue(std::locale(""));
	matrix<int> matrix1(3, 3, { {0,1,0}, {1,1,1}, {0,1,0} });
	matrix<int> matrix2(3, 4, { {1,1,1,1}, {1,0,0,1}, {1,1,1,1} });
	std::wcout << L"Test case 2:\n";
	print(std::wcout, kronecker_product(matrix1, matrix2));
}