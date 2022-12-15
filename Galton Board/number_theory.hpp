#pragma once
#include <concepts>
#include <vector>

template <typename T>
bool isPrime(const T& n)
	requires std::integral<T>
{
	// Corner cases
	if (n <= 1)
		return false;
	if (n <= 3)
		return true;

	// This is checked so that we can skip
	// middle five numbers in below loop
	if (n % 2 == 0 || n % 3 == 0)
		return false;

	// Using concept of prime number
	// can be represented in form of
	// (6*n + 1) or(6*n - 1) hence
	// we have to go for every multiple of 6 and
	// prime number would always be 1 less or 1 more then
	// the multiple of 6.
	for (T i = 5; i * i <= n; i = i + 6)
		if (n % i == 0 || n % (i + 2) == 0)
			return false;

	return true;
}

template <typename T>
T countPrimes(T strt, T end) {
	T count = 0;

	for (T i = strt; i <= end; i++)
		if (isPrime(i))
			count++;

	return count;
}

template <typename T>
T nthPrime(T n) {
	T counter = 0;
	T i;
	for (i = 2; counter < n; i++) {
		if (isPrime(i)) {
			counter++;
		}
	}
	return i - 1;
}

//OEIS A089175 a(n) = Floor[Sqrt[Prime[n]*PrimePi[n]]]
//Geometric mean of primes and their distribution function.
template <typename T>
std::vector<T> OEIS_A089175(T j) {  //j = 73
	std::vector<T> v;
	T k = 0, t = 0;
	for (T x = 1; x < j; x++) {
		if (isPrime(x)) t++;
		k = T(sqrt(nthPrime(x) * t));
		v.push_back(k);
		std::cout << x << " " << v.back() << std::endl;
	}
	return v;
}

//pi(n), the number of primes <= n.
template <typename T>
std::vector<T> OEIS_A000720(T j) {  //j = 73
	std::vector<T> v;
	T k = 0, t = 0;
	for (T x = 1; x <= j; x++) {
		if (isPrime(x)) t++;
		v.push_back(t);
		std::cout << x << " " << v.back() << std::endl;
	}
	return v;
}

//Numbers k such that k ^ 2 - k + 1 is prime.
template <typename T>
std::vector<T> OEIS_A055494(T j) { //j = 219
	std::vector<T> v;
	T k = 0;
	for (T k = 1; k < j; k++) {
		if (isPrime(T(pow(k, 2)) - k + 1)) {
			v.push_back(k);
			std::cout << k << " " << v.back() << std::endl;
		}
	}
	return v;
}

//Numbers k such that 2k + 1 is prime.
template <typename T>
std::vector<T> OEIS_A005097(T j) { //j = 219
	std::vector<T> v;
	T k = 0;
	for (T k = 1; k < j; k++) {
		if (isPrime(T(2 * k + 1))) {
			v.push_back(k);
			std::cout << k << " " << v.back() << std::endl;
		}
	}
	return v;
}

//Numbers k such that 2k - 1 is prime.
template <typename T>
std::vector<T> OEIS_A006254(T j) { //j = 219
	std::vector<T> v;
	T k = 0;
	for (T k = 1; k < j; k++) {
		if (isPrime(T(2 * k - 1))) {
			v.push_back(k);
			std::cout << k << " " << v.back() << std::endl;
		}
	}
	return v;
}

//Nearest-neighbors of odd primes, divided by 2. 
//OEIS_A005097 or OEIS_A006254
template <typename T>
std::vector<T> OEIS_A147820(T j) { //j = 219
	std::vector<T> v;
	T k = 0;
	for (T k = 1; k < j; k++) {
		if (isPrime(T(2 * k + 1)) || isPrime(T(2 * k - 1))) {
			v.push_back(k);
			std::cout << k << " " << v.back() << std::endl;
		}
	}
	return v;
}

template <typename T>
T OEIS_A048739(T n) {
	return T(floor(pow(1 + sqrt(2), n + 2) / 4));
}

template <typename T>
T binomial(T n, T k) {
	if (k == 0 || k == n)
		return 1;
	return binomial(n - 1, k - 1) + binomial(n - 1, k);
}

//a(n) = sum{ k = 0..floor(n / 2), binomial(n - k, k)(-1) ^ k * A048739(n - 2k) };
template <typename T>
std::vector<T>  OEIS_A099854(T j) {
	std::vector<T> v;
	for (T n = 0; n < j; n++) {
		T sum = 0;
		for (T k = 0; k <= (n / 2); k++)
			sum += T(binomial(n - k, k) * pow(-1, k) * OEIS_A048739(n - 2 * k));
		v.push_back(sum);
		std::cout << n << " " << v.back() << std::endl;
	}
	return v;
}

template <typename T>
	requires std::integral<T>
T prime_divisor_finder(const T& dividend);

template <typename T>
	requires std::integral<T>
T product_rule(const T& num1, const T& num2) {

	if (isPrime(num1) && isPrime(num2))
		return num1 + num2;
	else
		return num2 + prime_divisor_finder(num2) * num1;
}

template <typename T>
	requires std::integral<T>
T prime_divisor_finder(const T& dividend) {

	T res = 0;

	for (T prime_divisor = 2; prime_divisor < dividend; prime_divisor++)
	{
		if (isPrime(prime_divisor) && dividend % prime_divisor == 0)
			res = product_rule(prime_divisor, T(dividend / prime_divisor));
	}
	return res;
}

template <typename T>
	requires std::integral<T>
T derivative(const T& AD) {

	if (AD == 0 || AD == 1)
		return 0;

	else if (isPrime(AD))
		return 1;

	else if (AD % 2 == 0)
		return product_rule(2, T(AD / 2));

	else
		return prime_divisor_finder(AD);
}

template <typename T>
	requires std::integral<T>
T arithmetic_derivative(const T& AD) {
	T res = 0;

	if (AD >= 0)
		res = derivative(AD);  // D(-n) = -D(n)
	else
		res = -derivative(-AD);

	return res;
}

//a(n) = n' = arithmetic derivative of n: a(0) = a(1) = 0, a(prime) = 1, a(mn) = m*a(n) + n*a(m). 
template <typename T>
	requires std::integral<T>
void OEIS_A003415(const T n) {

	for (T i = 0; i < abs(n); i++) {
		if (n < 0)
			std::cout << i << " " << arithmetic_derivative(-i) << std::endl;
		else
			std::cout << i << " " << arithmetic_derivative(i) << std::endl;
	}
}

template <typename T>
	requires std::integral<T>
T Prime_omega_function(T n)
{
	T c = 2, count = 0;
	while (n > 1)
	{
		if (n % c == 0) {
			count++;
			//std::cout << c << " ";
			n /= c;
		}
		else c++;
	}
	return count;
}


