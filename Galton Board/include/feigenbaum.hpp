
template <typename elem, int order>
multicomplex<elem, order> Feigenbaum_constant
(
) //4.6692016091029906718532
{
	const int max_it = 14;
	const int max_it_j = 11;
	multicomplex<elem, order> a1 = 1.0, a2, d1 = 3.2, d, a, x, y;

	// std::cout << " i       d\n";
	for (int i = 2; i <= max_it; ++i) {
		a = a1 + (a1 - a2) / d1;
		for (int j = 1; j <= max_it_j; ++j) {
			x = 0;
			y = 0;
			for (int k = 1; k <= 1 << i; ++k) {
				y = 1 - 2 * y * x;
				x = a - x * x;
			}
			a -= x / y;
		}
		d = (a1 - a2) / (a - a1);
		// std::cout<< i << " " << d << '\n';
		d1 = d;
		a2 = a1;
		a1 = a;
	}

	return d;
}


template <typename elem, int order>
void Logistic_Map
(
	multicomplex<elem, order> z,
	int n
)
{
	multicomplex<elem, order> a = { 3.424,0 };//2.3 3.3 3.4

	static int i = 0;
	std::cout << i++ << " " << z << '\n';
	if (n == 0)return;
	z = a * z * sinh(1 - z);
	Logistic_Map(z, n - 1);
}