/// const operators, any scalar type

//---------------------------------------------------

template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
class multicomplex;

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const multicomplex<elem, order> operator+
(
	const T& a,
	const multicomplex<elem, order>& b
)
{
	const multicomplex<elem, order> n(static_cast<elem>(a));
	return n + b;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const multicomplex<elem, order> operator-
(
	const T& a,
	const multicomplex<elem, order>& b
)
{
	const multicomplex<elem, order> n(static_cast<elem>(a));
	return n - b;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const multicomplex<elem, order> operator*
(
	const T& a,
	const multicomplex<elem, order>& b
)
{
	const multicomplex<elem, order> n(static_cast<elem>(a));
	return b * n;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const multicomplex<elem, order> operator/
(
	const T& a,
	const multicomplex<elem, order>& b
)
{
	multicomplex<elem, order> n(a);
	return n / b;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const bool operator!=
(
	const multicomplex<elem, order>& a,
	const T& b
)
{
	const multicomplex<elem, order> n(static_cast<elem>(b));

	if (n.comp() == a.comp()) return false;
	return true;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const bool operator!=
(
	const T& a,
	const multicomplex<elem, order>& b
)
{
	const multicomplex<elem, order> n(static_cast<elem>(a));

	if (n.comp() == b.comp()) return false;
	return true;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const bool operator==
(
	const multicomplex<elem, order>& a,
	const T& b
)
{
	const multicomplex<elem, order> n(static_cast<elem>(b));

	if (n.comp() == a.comp()) return true;
	return false;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const bool operator==
(
	const T& a,
	const multicomplex<elem, order>& b
)
{
	const multicomplex<elem, order> n(static_cast<elem>(a));

	if (n.comp() == b.comp()) return true;
	return false;
}

//---------------------------------------------------

template <typename elem, int order>
inline const bool operator==
(
	const multicomplex<elem, order>& a,
	const multicomplex<elem, order>& b
)
{
	if (a.comp() == b.comp()) return true;
	return false;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const bool operator>
(
	const multicomplex<elem, order>& a,
	const T& b
)
{
	const multicomplex<elem, order> n(b);

	if (n.comp() > a.comp()) return true;
	return false;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const bool operator>=
(
	const multicomplex<elem, order>& a,
	const T& b
)
{
	const multicomplex<elem, order> n(b);

	if (n.comp() >= a.comp()) return true;
	return false;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const bool operator<
(
	const multicomplex<elem, order>& a,
	const T& b
)
{
	const multicomplex<elem, order> n(b);

	if (n.comp() < a.comp()) return true;
	return false;
}

//---------------------------------------------------

template <typename T, typename elem, int order>
inline const bool operator<=
(
	const multicomplex<elem, order>& a,
	const T& b
)
{
	const multicomplex<elem, order> n(b);

	if (n.comp() <= a.comp()) return true;
	return false;
}

//---------------------------------------------------

