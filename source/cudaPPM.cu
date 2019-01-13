#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>   
#include <stdlib.h>
#include <stdio.h> 

#include <cuda.h>

#define PI ((double)3.14159265358979) 
#define ALPHA ((double)0.7) // the alpha parameter of PPM

#define MAX(x, y) ((x > y) ? x : y)

#define BLOCK_SIZE 512
#define GRID_SIZE 2

// Halton sequence with reverse permutation
int primes[61] =
{
	2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,
	83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,
	191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283
};
__host__ __device__ inline int rev(const int i, const int p)
{
	if (i == 0)
	{
		return i;
	}
	else
	{
		return p - i;
	}
}

__host__ __device__ double hal(const int b, int j, int *primes)
{
	const int p = primes[b];
	double h = 0.0;
	double f = 1.0 / (double)p;
	double fct = f;
	while (j > 0)
	{
		h += rev(j % p, p) * fct;
		j /= p;
		fct *= f;
	}
	return h;
}

// Vector: position, also color (r,g,b)
class Vec3
{
public:
	__host__ __device__ Vec3(double x_ = 0, double y_ = 0, double z_ = 0)
	{
		x = x_;
		y = y_;
		z = z_;
	}
	__host__ __device__ inline Vec3 operator+(const Vec3 &b) const
	{
		return Vec3(x + b.x, y + b.y, z + b.z);
	}
	__host__ __device__ inline Vec3 operator-(const Vec3 &b) const
	{
		return Vec3(x - b.x, y - b.y, z - b.z);
	}
	__host__ __device__ inline Vec3 operator+(double b) const
	{
		return Vec3(x + b, y + b, z + b);
	}
	__host__ __device__ inline Vec3 operator-(double b) const
	{
		return Vec3(x - b, y - b, z - b);
	}
	__host__ __device__ inline Vec3 operator*(double b) const
	{
		return Vec3(x * b, y * b, z * b);
	}
	__host__ __device__ inline Vec3 mul(const Vec3 &b) const
	{
		return Vec3(x * b.x, y * b.y, z * b.z);
	}
	__host__ __device__ inline Vec3 normalized()
	{
		return (*this) * (1.0 / sqrt(x*x + y*y + z*z));
	}
	__host__ __device__ inline double dot(const Vec3 &b) const
	{
		return x * b.x + y * b.y + z * b.z;
	}
	__host__ __device__ Vec3 operator%(Vec3&b)
	{
		return Vec3(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
	}

	double x;
	double y;
	double z;
};

// Axis Aligned Bounding Box
class BoundingBox
{
public:
	inline void fit(const Vec3 &point)
	{
		if (point.x<min.x)min.x = point.x; // min
		if (point.y<min.y)min.y = point.y; // min
		if (point.z<min.z)min.z = point.z; // min
		max.x = MAX(point.x, max.x);
		max.y = MAX(point.y, max.y);
		max.z = MAX(point.z, max.z);
	}
	inline void reset()
	{
		min = Vec3(1e20, 1e20, 1e20);
		max = Vec3(-1e20, -1e20, -1e20);
	}

	Vec3 min;
	Vec3 max;
};

class HitPoint
{
public:
	Vec3 color;
	Vec3 position;
	Vec3 normal;
	Vec3 flux;
	double radius_squared;
	unsigned int n;
	int pixel;
	Vec3 BMin;
	Vec3 BMax;
};

class Photon
{
public:
	__host__ __device__ Photon()
	{
		hash_index = 0;
		intersection_point = 0;
		flux = 0;
		normal = 0;
	}

	unsigned int hash_index;
	Vec3 intersection_point;
	Vec3 flux;
	Vec3 normal;
};

class List
{
public:
	HitPoint *data;
	List *next;
};
List* ListAdd(HitPoint *item, List* head)
{
	List* p = new List;
	p->data = item;
	p->next = head;
	return p;
}

unsigned int num_hash;
unsigned int num_photon;
double hash_scale;
List *hitpoints = NULL;
BoundingBox hit_point_bbox;

HitPoint *hit_points_list = NULL;
Photon *photon_map = NULL;
unsigned int *total_count = 0;

// Spatial hash function
__host__ __device__ inline unsigned int hash(const int ix, const int iy, const int iz, const unsigned int num_hash)
{
	return (unsigned int)((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) % num_hash;
}

void build_hash_grid(const int width, const int height)
{
	// Find the bounding box of all the measurement points
	hit_point_bbox.reset();
	List *list = hitpoints;
	while (list != NULL)
	{
		HitPoint *hit_point = list->data;
		list = list->next;
		hit_point_bbox.fit(hit_point->position);
	}

	// Initial radius calculation
	Vec3 bbox_size = hit_point_bbox.max - hit_point_bbox.min;
	double initial_radius = ((bbox_size.x + bbox_size.y + bbox_size.z) / 3.0) / ((width + height) / 2.0) * 2.0 * 4.0;

	// Determine hash table size
	// Find the bounding box of all the measurement points, this time inflated by the initial radius
	hit_point_bbox.reset();
	list = hitpoints;
	int vphoton = 0;
	while (list != NULL)
	{
		HitPoint *hit_point = list->data;
		list = list->next;
		hit_point->radius_squared = initial_radius * initial_radius;
		hit_point->n = 0;
		hit_point->flux = Vec3();
		vphoton++;
		hit_point_bbox.fit(hit_point->position - initial_radius);
		hit_point_bbox.fit(hit_point->position + initial_radius);
	}

	// Make each grid cell two times larger than the initial radius
	hash_scale = 1.0 / (initial_radius*2.0);
	num_hash = vphoton;

	// Build the hash table
	photon_map = new Photon[num_hash];
	total_count = new unsigned int[num_hash];
	hit_points_list = new HitPoint[num_hash];

	list = hitpoints;
	int i = 0;
	while (list != NULL)
	{
		HitPoint *hit_point = list->data;
		Vec3 BMin = ((hit_point->position - initial_radius) - hit_point_bbox.min) * hash_scale;
		Vec3 BMax = ((hit_point->position + initial_radius) - hit_point_bbox.min) * hash_scale;
		hit_point->BMin = BMin;
		hit_point->BMax = BMax;
		hit_points_list[i++] = *hit_point;

		list = list->next;
	}
}

class Ray
{
public:
	__host__ __device__ Ray() {};
	__host__ __device__ Ray(Vec3 origin_, Vec3 direction_) : origin(origin_), direction(direction_) {}

	Vec3 origin;
	Vec3 direction;
};

// Material types
enum MaterialType
{
	DIFFUSE,
	SPECULAR,
	REFRACTIVE
};

class Sphere
{
public:
	Sphere(double radius_, Vec3 position_, Vec3 color_, MaterialType material_) :
		radius(radius_), position(position_), color(color_), material(material_) {}

	// Ray-Sphere intersection
	__host__ __device__ inline double intersect(const Ray &ray) const
	{
		Vec3 op = position - ray.origin;
		double t;
		double b = op.dot(ray.direction);
		double determinant = b*b - op.dot(op) + radius*radius;
		if (determinant < 0) {
			return 1e20;
		}
		else {
			determinant = sqrt(determinant);
		}
		t = b - determinant;
		if (t > 1e-4)
		{
			return t;
		}
		else
		{
			t = b + determinant;
			if (t > 1e-4)
			{
				return t;
			}
		}

		return 1e20;
	}

	double radius;
	Vec3 position;
	Vec3 color;
	MaterialType material;
};

Sphere spheres[] =
{ // Scene: radius, position, color, material
	Sphere(1e5, Vec3(1e5 + 1,40.8,81.6), Vec3(.75,.25,.25),DIFFUSE),//Left
	Sphere(1e5, Vec3(-1e5 + 99,40.8,81.6),Vec3(.25,.25,.75),DIFFUSE),//Right
	Sphere(1e5, Vec3(50,40.8, 1e5),     Vec3(.75,.75,.75),DIFFUSE),//Back
	Sphere(1e5, Vec3(50,40.8,-1e5 + 170), Vec3(),           DIFFUSE),//Front
	Sphere(1e5, Vec3(50, 1e5, 81.6),    Vec3(.75,.75,.75),DIFFUSE),//Bottomm
	Sphere(1e5, Vec3(50,-1e5 + 81.6,81.6),Vec3(.75,.75,.75),DIFFUSE),//Top
	Sphere(16.5,Vec3(27,16.5,47),       Vec3(1,1,1)*.999, SPECULAR),//Mirror
	Sphere(16.5,Vec3(73,16.5,88),       Vec3(1,1,1)*.999, REFRACTIVE),//Glass
	Sphere(8.5, Vec3(50,8.5,60),        Vec3(1,1,1)*.999, DIFFUSE) //Middle
};

// Tone mapping and gamma correction
int tone_map(double x)
{
	return int(pow(1 - exp(-x), 1 / 2.2) * 255 + .5);
}

// Find the closest intersection
__host__ __device__ inline bool intersect(const Ray &ray, double &t, int &object_id, Sphere* spheres)
{
	int N = 9; //sizeof(spheres) / sizeof(Sphere);
	double distance;
	const double infinity = 1e20;
	t = infinity;
	for (int i = 0; i < N; i++)
	{
		distance = spheres[i].intersect(ray);
		if (distance < t)
		{
			t = distance;
			object_id = i;
		}
	}
	return t < infinity;
}

// Generate a photon ray from the point light source with Quasi-Monte Carlo
__device__ void generate_photon(Ray* photon_ray, Vec3* flux, int photon_id, int *primes)
{
	*flux = Vec3(2500, 2500, 2500)*(PI*4.0);
	double p = 2.*PI*hal(0, photon_id, primes);
	double t = 2.*acos(sqrt(1. - hal(1, photon_id, primes)));
	double sint = sin(t);
	photon_ray->direction = Vec3(cos(p)*sint, cos(t), sin(p)*sint);
	photon_ray->origin = Vec3(50, 60, 85);
}

// HitPoint Pass
void eye_trace(const Ray &ray, int depth, const Vec3 &attenuation, unsigned int pixel_index)
{
	double t;
	int object_id;

	depth++;
	if (!intersect(ray, t, object_id, spheres) || (depth >= 20)) return;

	const Sphere &object = spheres[object_id];

	Vec3 intersection_point = ray.origin + ray.direction * t;
	Vec3 normal = (intersection_point - object.position).normalized();

	Vec3 base_color = object.color;

	// Lambertian
	if (object.material == DIFFUSE)
	{
		// Store the measurement point
		HitPoint* hit_point = new HitPoint;
		hit_point->color = base_color.mul(attenuation);
		hit_point->position = intersection_point;
		hit_point->normal = normal;
		hit_point->pixel = pixel_index;
		hitpoints = ListAdd(hit_point, hitpoints);
	}
	// Mirror
	else if (object.material == SPECULAR)
	{
		Ray reflection_ray(intersection_point, ray.direction - normal*2.0*normal.dot(ray.direction));
		eye_trace(reflection_ray, depth, base_color.mul(attenuation), pixel_index);
	}
	// Glass
	else
	{
		Vec3 nl = normal.dot(ray.direction) < 0 ? normal : normal * -1;

		Ray reflection_ray(intersection_point, ray.direction - normal*2.0*normal.dot(ray.direction));
		bool into = (normal.dot(nl)>0.0);

		double air_index = 1.0;
		double refractive_index = 1.5;

		double nnt = into ? air_index / refractive_index : refractive_index / air_index;

		double ddn = ray.direction.dot(nl);
		double cos2t = 1 - nnt*nnt*(1 - ddn*ddn);

		if (cos2t < 0)
		{
			return eye_trace(reflection_ray, depth, attenuation, pixel_index);
		}

		Vec3 refraction_direction = (ray.direction*nnt - normal*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).normalized();

		double a = refractive_index - air_index;
		double b = refractive_index + air_index;
		double R0 = a*a / (b*b);

		double cosinealpha = into ? -ddn : refraction_direction.dot(normal);
		double c = 1 - cosinealpha;

		double fresnel = R0 + (1 - R0)*c*c*c*c*c;
		Ray refraction_ray(intersection_point, refraction_direction);
		Vec3 attenuated_color = base_color.mul(attenuation);

		eye_trace(reflection_ray, depth, attenuated_color*fresnel, pixel_index);
		eye_trace(refraction_ray, depth, attenuated_color*(1.0 - fresnel), pixel_index);
	}
}

// Photon Pass
__global__ void photon_trace(Sphere *spheres, Photon *photon_map, unsigned int *total_count, int *primes, int photon_id, float hash_scale, Vec3 hit_point_bbox_min, unsigned int num_hash)
{
	double t;
	int object_id;

	unsigned int threadId = blockIdx.x*blockDim.x + threadIdx.x;
	photon_id += threadId;

	Ray ray;
	Vec3 flux;
	generate_photon(&ray, &flux, photon_id, primes);

	int depth = 0;
	Vec3 attenuation = Vec3(1, 1, 1);

	while (true)
	{
		depth++;
		int depth3 = depth * 3;

		if (!intersect(ray, t, object_id, spheres) || (depth >= 20))
		{
			break;
		}

		const Sphere &object = spheres[object_id];

		Vec3 intersection_point = ray.origin + ray.direction * t;
		Vec3 normal = (intersection_point - object.position).normalized();

		Vec3 base_color = object.color;
		double p = base_color.x > base_color.y && base_color.x > base_color.z ? base_color.x : base_color.y > base_color.z ? base_color.y : base_color.z;

		Vec3 nl = normal.dot(ray.direction) < 0 ? normal : normal * -1;

		// Lambertian
		if (object.material == DIFFUSE) 
		{
			// Use Quasi-Monte Carlo to sample the next direction
			double r1 = 2.*PI*hal(depth3 - 1, photon_id, primes);
			double r2 = hal(depth3 + 0, photon_id, primes);
			double r2s = sqrt(r2);

			Vec3 w = nl;
			Vec3 u = ((fabs(w.x) > .1 ? Vec3(0, 1) : Vec3(1)) % w).normalized();
			Vec3 v = w%u;
			Vec3 d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).normalized();

			// Compute hash index in for the hash grid and update current photon map
			Vec3 hh = (intersection_point - hit_point_bbox_min) * hash_scale;
			int ix = abs(int(hh.x));
			int iy = abs(int(hh.y));
			int iz = abs(int(hh.z));
			{
				Photon photon;

				photon.flux = flux;
				photon.hash_index = hash(ix, iy, iz, num_hash);
				photon.intersection_point = intersection_point;
				photon.normal = normal;

				photon_map[photon.hash_index] = photon;

				atomicAdd(&total_count[photon.hash_index], 1);
			}
			if (hal(depth3 + 1, photon_id, primes) < p)
			{
				ray = Ray(intersection_point, d);
				flux = base_color.mul(flux)*(1. / p);
			}
			else
			{
				break;
			}
		}
		// Mirror
		else if (object.material == SPECULAR)
		{
			ray = Ray(intersection_point, ray.direction - normal*2.0*normal.dot(ray.direction));
			flux = base_color.mul(flux);
			attenuation = base_color.mul(attenuation);
		}
		// Glass
		else
		{
			Ray reflection_ray(intersection_point, ray.direction - normal*2.0*normal.dot(ray.direction));
			bool into = (normal.dot(nl)>0.0);

			double air_index = 1.0;
			double refractive_index = 1.5;

			double nnt = into ? air_index / refractive_index : refractive_index / air_index;

			double ddn = ray.direction.dot(nl);
			double cos2t = 1 - nnt*nnt*(1 - ddn*ddn);

			if (cos2t < 0)
			{
				ray = reflection_ray;
			}
			else
			{
				Vec3 refraction_direction = (ray.direction*nnt - normal*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).normalized();

				double a = refractive_index - air_index;
				double b = refractive_index + air_index;
				double R0 = a*a / (b*b);

				double cosinealpha = into ? -ddn : refraction_direction.dot(normal);
				double c = 1 - cosinealpha;

				double fresnel = R0 + (1 - R0)*c*c*c*c*c;
				Ray refraction_ray(intersection_point, refraction_direction);
				Vec3 attenuated_color = base_color.mul(attenuation);

				double P = fresnel;

				// Photon ray (pick one via Russian roulette)
				if (hal(depth3 - 1, photon_id, primes) < P)
				{
					ray = reflection_ray;
				}
				else
				{
					ray = refraction_ray;
					attenuation = attenuated_color;
				}
			}
		}
	}
}

__global__ void accumulate_photons(Photon *photon_map, unsigned int *total_count, HitPoint *hit_points, int total_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < total_size)
	{
		HitPoint *hit_point = &hit_points[idx];

		for (int iz = abs(int(hit_point->BMin.z)); iz <= abs(int(hit_point->BMax.z)); iz++)
		{
			for (int iy = abs(int(hit_point->BMin.y)); iy <= abs(int(hit_point->BMax.y)); iy++)
			{
				for (int ix = abs(int(hit_point->BMin.x)); ix <= abs(int(hit_point->BMax.x)); ix++)
				{
					int hash_index = hash(ix, iy, iz, total_size);

					Vec3 flux = photon_map[hash_index].flux * total_count[hash_index];// *(2.0 / GRID_SIZE);
					Vec3 v = hit_point->position - photon_map[hash_index].intersection_point;
					if ((hit_point->normal.dot(photon_map[hash_index].normal) > 1e-3) && (v.dot(v) <= hit_point->radius_squared)) {
						// unlike N in the paper, hit_point->n stores "N / ALPHA" to make it an integer value
						double radius_reduction = (hit_point->n * ALPHA + ALPHA) / (hit_point->n * ALPHA + 1.0);
						hit_point->radius_squared = hit_point->radius_squared * radius_reduction;
						hit_point->n++;
						hit_point->flux = (hit_point->flux + hit_point->color.mul(flux)*(1. / PI))*radius_reduction;
					}
				}
			}
		}
	}
}

int main(int argc, char *argv[]) {
	int width = 1024;
	int height = 768;
	int samples = (argc == 2) ? MAX(atoi(argv[1]) / 1000, 1) : 1000;

	Vec3 ray_origin = Vec3(50, 48, 295.6);
	Vec3 ray_direction = Vec3(0, -0.042612, -1);
	ray_direction = ray_direction.normalized();

	Ray camera(ray_origin, ray_direction);

	Vec3 cx = Vec3(width * 0.5135 / height);
	Vec3 cy = (cx % camera.direction).normalized() * 0.5135;
	Vec3 *color = new Vec3[width * height];
	Vec3 vw;

	// Trace eye rays and store measurement points
	for (int y = 0; y < height; y++)
	{
		fprintf(stderr, "\rHitPointPass %5.2f%%", 100.0*y / (height - 1));

		for (int x = 0; x < width; x++)
		{
			unsigned int pixel_index = x + y * width;
			Vec3 direction = cx * ((x + 0.5) / width - 0.5) + cy * (-(y + 0.5) / height + 0.5) + camera.direction;
			Ray ray = Ray(camera.origin + direction * 140, direction.normalized());

			eye_trace(ray, 0, Vec3(1, 1, 1), pixel_index);
		}
	}
	fprintf(stderr, "\n");

	// Build the hash table over the measurement points
	build_hash_grid(width, height);

	// Trace photon rays with CUDA
	Sphere *spheres_d;
	Photon *photon_map_d;
	unsigned int *total_count_d;
	int *primes_d;
	HitPoint *hit_points_d;

	dim3 grid;
	dim3 grid2;
	dim3 block;

	block.x = BLOCK_SIZE;
	grid.x = GRID_SIZE;
	grid2.x = (num_hash + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaMalloc((void **)&spheres_d, sizeof(spheres));
	cudaMalloc((void **)&photon_map_d, num_hash * sizeof(Photon));
	cudaMalloc((void **)&total_count_d, num_hash * sizeof(unsigned int));
	cudaMalloc((void **)&primes_d, sizeof(primes));
	cudaMalloc((void **)&hit_points_d, num_hash * sizeof(HitPoint));

	cudaMemcpy(spheres_d, spheres, sizeof(spheres), cudaMemcpyHostToDevice);
	cudaMemcpy(primes_d, primes, sizeof(primes), cudaMemcpyHostToDevice);
	cudaMemcpy(hit_points_d, hit_points_list, num_hash * sizeof(HitPoint), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	num_photon = samples;
	for (unsigned int i = 0; i < num_photon; i++)
	{
		{
			cudaDeviceSynchronize();
			cudaMemset(photon_map_d, 0, num_hash * sizeof(Photon));
			cudaMemset(total_count_d, 0, num_hash * sizeof(unsigned int));
		}

		double percentage = 100.0 * (i + 1) / num_photon;
		fprintf(stderr, "\rPhotonPass %5.2f%%", percentage);
		int m = BLOCK_SIZE * GRID_SIZE * i;

		photon_trace << < grid, block >> > (spheres_d, photon_map_d, total_count_d, primes_d, m, hash_scale, hit_point_bbox.min, num_hash);

		cudaDeviceSynchronize();

		accumulate_photons << <grid2, block >> > (photon_map_d, total_count_d, hit_points_d, num_hash);
	}
	fprintf(stderr, "\n");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);

	fprintf(stderr, "Work done in %f ms", time);

	cudaMemcpy(hit_points_list, hit_points_d, num_hash * sizeof(HitPoint), cudaMemcpyDeviceToHost);

	cudaFree(spheres_d);
	cudaFree(photon_map_d);
	cudaFree(total_count_d);
	cudaFree(primes_d);

	// Density estimation
	for (int index = 0; index < num_hash; index++)
	{
		HitPoint hit_point = hit_points_list[index];
		int i = hit_point.pixel;
		color[i] = color[i] + hit_point.flux*(1.0 / (PI*hit_point.radius_squared*num_photon* BLOCK_SIZE * GRID_SIZE));
	}

	// Save the image after tone mapping and gamma correction
	FILE* f = fopen("image.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	for (int i = 0; i< width * height; i++)
	{
		fprintf(f, "%d %d %d ", tone_map(color[i].x), tone_map(color[i].y), tone_map(color[i].z));
	}
}