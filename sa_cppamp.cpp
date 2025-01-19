#include <iostream>
#include <vector>
#include <array>
#include <amp.h>  // C++ AMP header
#include <amp_math.h>  // C++ AMP header
#include <cmath>
#include <random>
#include <chrono>

using namespace concurrency; // Namespace for C++ AMP


void fill_with_random(std::vector<double>& out, double left, double right)
{
	const unsigned int n = out.size();
	std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(left, right);
	for(unsigned int i =0; i < n; ++i)
	{
		out[i] = dis(gen);
	}
}

double get_random()
{
	std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
	return dis(gen);
}

void compute_rastrigin_on_gpu(const std::vector<double>& input, std::vector<double>& output)
{
	const int out_size = output.size();
	const int in_size = input.size();

    // Copy data to an accelerator_view using array_view
    array_view<const double, 1> input_view(in_size, input);
    array_view<double, 1> output_view(out_size, output);
	
	const unsigned int n = in_size/out_size;

    // Mark output_view as writeable
    output_view.discard_data();

    // Launch computation on the GPU
    parallel_for_each(
        output_view.extent, // Defines the range of indices
        [=](index<1> idx) restrict(amp) 
		{
			double result = 0;
			for(unsigned int i = 0; i < n; ++i)
			{
				auto idx_copy = idx;
				idx_copy*=n;
				idx_copy+=i;
				result += fast_math::pow(input_view[idx_copy],2) - 10 * fast_math::cos(2 * M_PI * input_view[idx_copy]);
			}
			
			output_view[idx] = result + 10 * n;
        });

    // Synchronize the data back to the CPU
    output_view.synchronize();
};

void compute_quartic_on_gpu(const std::vector<double>& input, std::vector<double>& output)
{
	const int out_size = output.size();
	const int in_size = input.size();

    array_view<const double, 1> input_view(in_size, input);
    array_view<double, 1> output_view(out_size, output);
	
	const unsigned int n = in_size/out_size;

    output_view.discard_data();

    parallel_for_each(
        output_view.extent,
        [=](index<1> idx) restrict(amp) 
		{
			double result = 0;
			for(unsigned int i = 0; i < n; ++i)
			{
				auto idx_copy = idx;
				idx_copy*=n;
				idx_copy+=i;
				result += fast_math::pow(input_view[idx_copy],2) * i;
			}
			output_view[idx] = fast_math::pow(result,2);
        });

    output_view.synchronize();
};

void compute_rosenbrock_on_gpu(const std::vector<double>& input, std::vector<double>& output)
{
	const int out_size = output.size();
	const int in_size = input.size();

    array_view<const double, 1> input_view(in_size, input);
    array_view<double, 1> output_view(out_size, output);
	
	const unsigned int n = in_size/out_size;

    output_view.discard_data();

    parallel_for_each(
        output_view.extent, 
        [=](index<1> idx) restrict(amp) 
		{
			double result = 0;
			for(unsigned int i = 1; i < n; ++i)
			{
				auto idx_copy = idx;
				idx_copy*=n;
				idx_copy+=i;
				auto idx_copy_2 = idx_copy;
				idx_copy_2 -= 1;
								
				result += 100 * fast_math::pow(input_view[idx_copy] - fast_math::pow(input_view[idx_copy_2],2),2) + fast_math::pow(1 - input_view[idx_copy_2], 2);
			}
			output_view[idx] = 1+ result;
        });

    output_view.synchronize();
};

template<typename F>
double simulated_annealing_sequential(F func, unsigned int iterations, double temperature, double alpha, double eps, double left_limit, double right_limit, const std::vector<double>& initial,std::vector<double>& out)
{
	std::vector<double> current_x = initial;
	double current_result  = func(current_x);
	std::vector<double> new_x(initial.size());
	out = initial;
	
	while(temperature > eps)
	{
		for(unsigned int i = 0; i < iterations; ++i)
		{
			fill_with_random(new_x, left_limit, right_limit);
			double new_result = func(new_x);
			
			if(new_result < current_result)
			{
				current_result = new_result;
				current_x = new_x;
				out = new_x;
			}
			else
			{
				const double r = get_random();
                const double E = exp( (current_result - new_result) / temperature );
                if (r < E)
                {
                    current_x = new_x; 
					current_result = new_result;
                }
			}
		}
		
		temperature = temperature * (1 - alpha);
	}
	
	return func(out);	
}

template<typename F, typename G>
double simulated_annealing_gpu(F func,G gpu_func, unsigned int iterations, double temperature, double alpha, double eps, double left_limit, double right_limit, const std::vector<double>& initial,std::vector<double>& out)
{
	const auto N = initial.size();
	std::vector<double> current_x = initial;	
	double current_result  = func(current_x);
	std::vector<double> results_of_whole_iteration(iterations);
	std::vector<double> solutions_of_whole_iteration(iterations * N);
	out = initial;
	
	while(temperature > eps)
	{		
		fill_with_random(solutions_of_whole_iteration, left_limit, right_limit);
		gpu_func(solutions_of_whole_iteration,results_of_whole_iteration);
		
		for(unsigned int i = 0; i < iterations; ++i)
		{
			double new_result = results_of_whole_iteration[i];
						
			if(new_result < current_result)
			{
				current_result = new_result;
				current_x = std::vector<double>(std::begin(solutions_of_whole_iteration) + i * N, std::begin(solutions_of_whole_iteration) + i * N + N);
				out = current_x;
			}
			else
			{
				const double r = get_random();
                const double E = exp( (current_result - new_result) / temperature );
                if (r < E)
                {
                    current_x = std::vector<double>(std::begin(solutions_of_whole_iteration) + i * N, std::begin(solutions_of_whole_iteration) + i * N + N);
					current_result = new_result;
                }
			}
		}
		temperature = temperature * (1 - alpha);
	}
	
	return func(out);	
}

int main(int argc,char** argv) 
{
	using clock = std::chrono::system_clock;
	using sec = std::chrono::duration<double>;
	
	const unsigned int size_of_task = std::stoul(argv[4]);
	const unsigned int test_iterations = std::stoul(argv[2]);
	const std::string function_name = argv[3];
	const unsigned int loop_iterations = std::stoul(argv[1]);
	
	if(argc < 4)
	{
		std::cout << "Usage: sa_cppamp <loop iterations> <test iterations> <function(quartic|rastrigin|rosenbrock)> <size of task>\n";
		return 0;
	}
	
	std::vector<accelerator> accs = accelerator::get_all();

    for (int i = 0; i <accs.size(); i++) 
	{
		std::cout << "Accelerator: " << i << '\n';
        std::wcout << accs[i].device_path << "\n";
        std::wcout << accs[i].dedicated_memory << "\n";
        std::wcout << (accs[i].supports_cpu_shared_memory ?
            "CPU shared memory: true" : "CPU shared memory: false") << "\n";
        std::wcout << (accs[i].supports_double_precision ?
            "double precision: true" : "double precision: false") << "\n";
        std::wcout << (accs[i].supports_limited_double_precision ?
            "limited double precision: true" : "limited double precision: false") << "\n\n";		
    }
	
	std::cout << "## Simulated annealing on GPU using C++AMP, size of task: " << size_of_task << ",test iterations: " << test_iterations << ", on: " << function_name << " function " << "with L set in simulated annealing to: " << loop_iterations << "\n";
	
	auto compute_rastrigin = [](const std::vector<double>& in)
	{
		const unsigned int n = in.size();
		double result = 0;
		for(auto x : in)
		{
			result += std::pow(x,2) - 10 * std::cos(2 * M_PI * x);
		}
		
		return result + 10 * n;
	};
	
	auto compute_quartic = [](const std::vector<double>& in)
	{
		const unsigned int n = in.size();
		double result = 0;
		for(unsigned int i =0; i < n; ++i)
		{
			result += i * std::pow(in[i],2);
		}
		
		return std::pow(result,2);
	};
	
	auto compute_rosenbrock = [](const std::vector<double>& in)
	{
		const unsigned int n = in.size();
		double result = 0;
		for(unsigned int i = 1; i < n; ++i)
		{
			result += 100 * std::pow(in[i] - std::pow(in[i-1],2), 2) + std::pow(1 - in[i-1], 2);
		}
		
		return 1 + result;
	};
	
	std::function<void(const std::vector<double>&, std::vector<double>&)> function_gpu;
	std::function<double(const std::vector<double>&)> function;
	double initial_value = 0;
	double left_limit = 0;
	double right_limit = 0;
	
	if(function_name == "rastrigin")
	{
		function_gpu = compute_rastrigin_on_gpu;
		function = compute_rastrigin;
		initial_value = 4.65;
		left_limit = -5.12;
		right_limit = 5.12;
	}
	else if(function_name == "quartic")
	{
		function_gpu = compute_quartic_on_gpu;
		function = compute_quartic;
		initial_value = 1.0;
		left_limit = -1.1;
		right_limit = 1.1;
	}
	else if(function_name == "rosenbrock")
	{
		function_gpu = compute_rosenbrock_on_gpu;
		function = compute_rosenbrock;
		initial_value = 1.0/(size_of_task + 1.0);
		left_limit = -3;
		right_limit = 3;
	}
	else
	{
		std::cout << "Usage: sa_cppamp <size of task> <test iterations> <function(quarti|rastrigin)>\n";
		return 0;
	}
	
	const unsigned int L = loop_iterations;
	const double T = 1000;
	const double alpha = 0.01;
	const double eps = 0.00001;
	
	unsigned int sequential_test_time = 0;
	
	for(unsigned int i = 0; i < test_iterations ; ++i)
	{
		std::vector<double> initial_x(size_of_task,initial_value);
		std::vector<double> optimal_x(size_of_task);
		
		auto before = clock::now();
		
		auto sa_sequential_result = simulated_annealing_sequential(function,L,T,alpha,eps,left_limit, right_limit, initial_x,optimal_x);
		
		std::cout << "Result of simulated annealing using sequential approach: " << sa_sequential_result << std::endl;
		
		for(unsigned int i = 0; i < optimal_x.size(); ++i)
		{
			std::cout << " x[" << i << "] = " << optimal_x[i];
		}
		std::cout << '\n';
		
		sec duration = clock::now() - before;
		sequential_test_time += duration.count();
	}
	
	unsigned int sequential_avg_test_time = sequential_test_time/test_iterations;
	
	std::cout << "Sequential took avg: " << sequential_avg_test_time << "s" << std::endl;
	
	unsigned int gpu_test_time = 0;
	
	for(unsigned int i = 0; i < test_iterations ; ++i)
	{
		std::vector<double> initial_x(size_of_task,initial_value);
		std::vector<double> optimal_x(size_of_task);
		
		auto before = clock::now();
		
		auto sa_gpu_result = simulated_annealing_gpu(function,function_gpu,L,T,alpha,eps,left_limit, right_limit, initial_x,optimal_x);
		
		std::cout << "Result of simulated annealing using GPU via C++AMP: " << sa_gpu_result << std::endl;
		
		for(unsigned int i = 0; i < optimal_x.size(); ++i)
		{
			std::cout << "x[" << i << "] = " << optimal_x[i];
		}
		
		sec duration = clock::now() - before;
		gpu_test_time += duration.count();
	}
	
	unsigned int gpu_avg_test_time = gpu_test_time/test_iterations;
	
	std::cout << "GPU took avg: " << gpu_avg_test_time << "s" << std::endl;
	
	double speed_up = static_cast<double>(sequential_avg_test_time) / static_cast<double>(gpu_avg_test_time);
	
	std::cout << "speed_up using GPU: " << speed_up << std::endl;
		
    return 0;
}
