#pragma once

#include <iostream>
#include <coroutine>

//https://nmilo.ca/blog/coroutines.html

template <class T>
struct Generator {
  struct promise_type {
    T value;
    bool finished;

    Generator get_return_object() { return { std::coroutine_handle<promise_type>::from_promise(*this) }; }

    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void unhandled_exception() {}

    std::suspend_always yield_value(T val) { value = val; return {}; }
    void return_void() { finished = true; }
  };

  std::coroutine_handle<promise_type> handle;

  struct Sentinel {};
  struct Iterator {
    Generator& g;

    T operator*() const {
      return g.handle.promise().value;
    }

    Iterator& operator++() {
      g.handle.resume();
      return *this;
    }

    friend bool operator==(const Iterator& it, Sentinel) { return it.g.handle.promise().finished; }
    friend bool operator!=(const Iterator& it, Sentinel) { return !it.g.handle.promise().finished; }
  };

  Iterator begin() { return { *this }; }
  Sentinel end() { return {}; }
};

template <class T>
Generator<T> get_squares(const T max) {
  for (int i = 0; i < max; i++) {
    co_yield i * i;
  }
}

template <class T>
void cor_driver(const T& in) {
  for (auto square : get_squares(in)) {
    std::cout << square << std::endl;
  }
}