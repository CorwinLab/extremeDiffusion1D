#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <assert.h>

template<class Temp>
void print_generic(Temp vec) {
	// Prints an object with an interator member.
	// Should cut this off at like 10 items but w/e
	std::cout << "[";
	for (auto i = vec.begin(); i != vec.end(); i++) {
		std::cout << *i << ", ";
	}
	std::cout << "] \n";
};

void myfunction(std::vector<int>& vec){
  for (unsigned int i = 0; i < vec.size(); i++){
    int* vec_val = &vec.at(i);
    *vec_val = i;
  }
}

int main(){
  std::vector<int> vec = {9, 6, 7};
  myfunction(vec);
  print_generic(vec);

  /*
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
  bool statement = false;
  assert(statement && "Whatever the error is");
  */

  return 0;
}
