#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <limits.h>
#include <omp.h>
#include <math.h>

using namespace std;
using glm::vec3;
using glm::mat3;

class Light
{
public:
	glm::vec3 lightPos;
	glm::vec3 lightColor;
	float intensity;

	Light(glm::vec3 lightPos, glm::vec3 lightColor, float intensity)
	 : lightPos(lightPos), lightColor(lightColor), intensity(intensity){}
	 Light(){}
};

//Screen Initialisation
const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
SDL_Surface* screen;
int t;

//Camera Initialisation
float focalLength = 250.0f;
vec3 cameraPos(0.0f, 0.0f, -1.7f);
mat3 R = mat3(0.0f);
float yaw = 0.0;

//Lights initialisation
int numLights = 0;
Light lights[20];
vec3 indirectLight = 0.5f*vec3(1,1,1);
vec3 randomPositions[180];
vec3 pixels[SCREEN_WIDTH * SCREEN_HEIGHT];

//Declare the triangles and intersections
vector<Triangle> triangles;
struct Intersection
{
	vec3 position;
	float distance;
	int triangleIndex;
};
vector<Intersection> closestIntersection;

//Buttons
bool antiAliasing_key = false;
bool shadows_key = false;
bool darken_key = false;
bool illuminate_key = false;
bool updated = true;

//Antialiasing
bool antiaAliasing = false;
int antiaAliasing_samples = 4;

//Soft shadows
bool softShadows = false;
int softShadows_samples = 9;

/*--------------------------------------------------FUNCTIONS-------------------------------------------------------------------------*/
void Illuminate(vec3 position, vec3 color, float intensity)
{

	lights[numLights].lightPos = position;
	lights[numLights].lightColor = color;
	lights[numLights].intensity = intensity;

	for(int i = 0; i < softShadows_samples; i++)
	{
		vec3 randomPos(lights[numLights].lightPos.x + ((((double) rand() / (RAND_MAX)) - 0.5f) * 0.08f), lights[numLights].lightPos.y + ((((double) rand() / (RAND_MAX)) - 0.5f) * 0.08f), lights[numLights].lightPos.z + ((((double) rand() / (RAND_MAX)) - 0.5f) * 0.08f));
		randomPositions[(numLights*softShadows_samples) + i] = randomPos;
	}

	numLights++;
	
}

bool ClosestIntersection(vec3 start, vec3 dir, const vector<Triangle>& triangles,
						 Intersection& closestIntersection, bool lighted, int x, int y)
{
	//Initialise the returned boolean
	bool intersection = false;
	
	//test all the created triangles for intersections
	for (size_t i = 0; i < triangles.size(); i++)
	{
		//3D vectors representing the vertices of the triangle
		vec3 v0 = triangles[i].v0;
		vec3 v1 = triangles[i].v1;
		vec3 v2 = triangles[i].v2;

		//Set the edges to v0 to be the origin of the coordinate system
		vec3 e1 = v1 - v0;
		vec3 e2 = v2 - v0;

		//Start computing the components of the 3x3 linear system
		vec3 b = start - v0;

		//Apply Cramers rule for a faster solution. Necessary to compute the cross product of all the collums of A matrix.
		vec3 det1=glm::cross(e1,e2);
		vec3 det2=glm::cross(b,e2);
		vec3 det3=glm::cross(e1,b);
		
		vec3 A1=-dir;

		//Compute the Cramers rule's sud-determinants
		float sol1=det1.x*b.x+det1.y*b.y+det1.z*b.z;
		float sol2=det2.x*A1.x+det2.y*A1.y+det2.z*A1.z;
		float sol3=det3.x*A1.x+det3.y*A1.y+det3.z*A1.z;

		//Compute the determinant of A
		float det=det1.x*A1.x+det1.y*A1.y+det1.z*A1.z;

		//Find the solutions of the 3x3 system by dividing sol1,sol2,sol3 with the determinant.
		float t=sol1/det;
		float u=sol2/det;
		float v=sol3/det;

		//Points need to be both on the same plane and within the triangle
		if(u+v<=1.0f && u>=0.0f && v>=0.0f && t>=0.0f)//CHECK INEQUALITIES 7,8,9,11
		{
			//describe point n in the triangle
			vec3 r=v0+(u*e1)+(v*e2);
			float pos_distance=glm::distance(start,r);

			//The distance needs to be the minimum value calculated.
			if(closestIntersection.distance>=pos_distance)
			{
				//If that condition holds the interesections structure needs to be updated
				closestIntersection.position=r;
				closestIntersection.distance=pos_distance;
				closestIntersection.triangleIndex=i;
				
			}
			intersection = true;
		}
	}


	return intersection;
}


vec3 DirectLight(const Intersection& intersection)
{
	
	int samples;
	vec3 inner(0.0f,0.0f,0.0f);
	vec3 final(0,0,0);

	if(softShadows)
		samples = softShadows_samples;
	else
		samples = 1;

	for(int k = 0; k < numLights; k++)
	{
		for(int counter = 0; counter < samples; counter++)
		{
			vec3 position;
			vec3 lightColor = lights[k].lightColor * lights[k].intensity;

			if(samples != 1)
			{
				position = randomPositions[(k*softShadows_samples) + counter];
			}
			else
			{
				position = lights[k].lightPos;
			}

			
			float r = glm::distance(intersection.position, position);
			float A = 4*M_PI*pow(r,2);
			vec3 P = lightColor /= (float) samples;
			vec3 rVec = glm::normalize(position - intersection.position);
			vec3 nVec = glm::normalize(triangles[intersection.triangleIndex].normal);
			vec3 B = P/A;
			vec3 D = B * max(glm::dot(rVec,nVec), 0.0f);

			//To compute the direct shadows check with closestIntersection
			Intersection j;
			j.distance = std::numeric_limits<float>::max();
			bool isIntersection=ClosestIntersection(position, -rVec, triangles, j, true, 0, 0);
			if (isIntersection)
			{
				// if intersection is closer to light source than self
				if (j.distance < r*0.99f) // small multiplier to reduce noise
					D = vec3 (0.0f, 0.0f, 0.0f);
			}

			// the color stored in the triangle is the reflected fraction of light
			inner = inner + D;
		}
		
		final = final + inner;
	}

	vec3 p = triangles[intersection.triangleIndex].color;
	return final*p;
}

void Update()
{
	//Compute frame time
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	//Start again by reseting the intersection distances
	float m = std::numeric_limits<float>::max();
	for(int i = 0; i < SCREEN_WIDTH*SCREEN_HEIGHT; i++)
	{
		//Set all the intersection distances to the maximum possible float number
		closestIntersection[i].distance = m;
	}

	//Adjust the camera rotation matrix
	vec3 right(R[0][0], R[0][1], R[0][2]);
	vec3 down(R[1][0], R[1][1], R[1][2]);
	vec3 forward(R[2][0], R[2][1], R[2][2]);

	//Initialise all keys to be pressed
	Uint8* keystate = SDL_GetKeyState( 0 );
	
	//Camera moves forward
	if( keystate[SDLK_UP] )
	{
		
		cameraPos = cameraPos + 0.1f*forward;
		updated = true;
	}//Camera moves backwards
	else if( keystate[SDLK_DOWN] )
	{
		
		cameraPos = cameraPos - 0.1f*forward;
		updated = true;
	}

	//Camera rotates to the left
	if( keystate[SDLK_LEFT] )
	{
		
		yaw = yaw + 0.1f;
		updated = true;
	}//Camera rotates to the right
	else if( keystate[SDLK_RIGHT] )
	{
		
		yaw = yaw - 0.1f;
		updated = true;
	}


	// Lights movement 
	if (keystate[SDLK_w])
	{
		lights[0].lightPos = lights[0].lightPos + 0.1f*forward;
		for(int i = 0; i < softShadows_samples; i++)
		{
			randomPositions[i] = randomPositions[i] + 0.1f*forward;
		}
		updated = true;
	}
	else if (keystate[SDLK_s])
	{
		lights[0].lightPos = lights[0].lightPos - 0.1f*forward;
		for(int i = 0; i < softShadows_samples; i++)
		{
			randomPositions[i] = randomPositions[i] - 0.1f*forward;
		}
		updated = true;
	}

	if (keystate[SDLK_a])
	{
		lights[0].lightPos = lights[0].lightPos - 0.1f*right;
		for(int i = 0; i < softShadows_samples; i++)
		{
			randomPositions[i] = randomPositions[i] - 0.1f*right;
		}
		updated = true;
	}
	else if (keystate[SDLK_d])
	{
		lights[0].lightPos = lights[0].lightPos + 0.1f*right;
		for(int i = 0; i < softShadows_samples; i++)
		{
			randomPositions[i] = randomPositions[i] + 0.1f*right;
		}
		updated = true;
	}

	//Enable anti anti aliasing to reduce the staircase effect
	if( antiAliasing_key && keystate[SDLK_1])
	{
		antiaAliasing = !antiaAliasing;
		cout << "Antialiasing" << antiaAliasing << endl;
	 	antiAliasing_key = true;
		updated = true;
	}
	else if (!keystate[SDLK_1])
	{
	 antiAliasing_key = false;
	}
    

    //Enable soft shadow to smooth the casted shadows
	if(!shadows_key && keystate[SDLK_2])
	{
		softShadows = !softShadows;
		cout << "Soft Shadows " << softShadows << endl;
		shadows_key = true;
		updated = true;
	}
	else if (!keystate[SDLK_2])
	{
		shadows_key = false;
	}

	//Increase the number of lights
	if(!illuminate_key && keystate[SDLK_l])
	{
		vec3 pos = vec3((((double) rand() / (RAND_MAX)) - 0.5f) * 2.0f, (((double) rand() / (RAND_MAX)) - 0.5f) * 2.0f, (((double) rand() / (RAND_MAX)) - 0.5f) * 2.0f);
		vec3 color = vec3(abs((((double) rand() / (RAND_MAX)) - 0.5f)) * 2.0f + 0.2f,abs((((double) rand() / (RAND_MAX)) - 0.5f)) * 2.0f + 0.2f,abs(((double) rand() / (RAND_MAX)) - 0.5f) * 2.0f + 0.2f);
		float intensity = abs(((double) rand() / (RAND_MAX)) - 0.5f) * 20.0f;
		Illuminate(pos, color , intensity);
		cout << "More lights" << endl;
		illuminate_key = true;
		updated = true;
	}
	else if (!keystate[SDLK_l])
	{
		illuminate_key = false;
	}

	//Decrease the number of lights
	if(!darken_key && keystate[SDLK_b])
	{
		if(numLights > 0)
			numLights--;
		cout << "Less lights" << endl;
		darken_key = true;
		updated = true;
	}
	else if (!keystate[SDLK_b])
	{
		darken_key = false;
	}
		
	// Update camera rotation matrix
	float c = cos(yaw);
	float s = sin(yaw);
	R[0][0] = c;
	R[0][2] = s;
	R[2][0] = -s;
	R[2][2] = c;

}

void Draw()
{
	//Loop through all the pixels and compute the corresponding ray direction.
	//Call closestIntersection to get the closest intersection in that direction.
	//If there is an intersection the color of the pixel should be set to the color of that triangle
	
	int samples; //If antiAliasing is true then this variable will be the number of samples that will be used to prevent aliasing

	if(antiaAliasing)
	{
		samples = antiaAliasing_samples;
	}

	else
	{
		samples = 1;
	}
		

	// This is the loop that needs parallelisation
	#pragma omp parallel 
	{

	 #pragma omp for nowait
	 for (int y = 0; y < SCREEN_HEIGHT; y++)
	 {
		float x1, y1;

		for (int x = 0; x < SCREEN_WIDTH; x++)
		{
			vec3 colorCollector(0.0f,0.0f,0.0f);
			if(samples > 1) 
				y1 = y - 0.3f;
			else
				y1 = y;

			for(int alias1 = 0; alias1 < samples; alias1++)
			{
				if(samples > 1) 
					x1 = x - 0.3f;
				else
					x1 = x;

				for(int alias2 = 0; alias2 < samples; alias2++)
				{
					//compute the corresponding ray direction
					vec3 d(x1-(float)SCREEN_WIDTH/2.0f, y1 - (float)SCREEN_HEIGHT/2.0f, focalLength);
					
					if (ClosestIntersection(cameraPos, R*d, triangles, closestIntersection[y*SCREEN_HEIGHT + x], false, x, y ))
					{
						//Compute both direct light and indirect light

						vec3 color = DirectLight(closestIntersection[y*SCREEN_HEIGHT+x]);
						vec3 D = color;
						vec3 N = indirectLight;
						vec3 T = D + N;
						vec3 p = triangles[closestIntersection[y*SCREEN_HEIGHT+x].triangleIndex].color;
						vec3 R = p*T;

						// direct shadows cast to point from light
						colorCollector= colorCollector+R;

						//change the value of x1 for antialiasing
						x1 = x1+(1.0f / (float) (samples - 1));
					}
				}
				//change the value of y1 for antialiasing
				y1 = y1 + (1.0f / (float) (samples - 1));
			}

			colorCollector = colorCollector /( (float)(pow(samples,2)));
			pixels[y*SCREEN_HEIGHT + x] = colorCollector;

		}
	 }
	}
		
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	// Spawn threads

	#pragma omp parallel
	{	
		#pragma omp for nowait
		for (int y = 1; y < SCREEN_HEIGHT - 1; y++)
		{
			for (int x = 1; x < SCREEN_WIDTH - 1; x++)
			{	
				vec3 finalColour(0.0f,0.0f,0.0f);
				finalColour = pixels[y*SCREEN_HEIGHT+x];

				PutPixelSDL( screen, x, y, finalColour );
			}
		}
	}
	

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);
		

	SDL_UpdateRect( screen, 0, 0, 0, 0 );

}

int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	Illuminate(vec3(0, -0.5f, -0.7f), vec3(1,1,1), 14 );

	// Request as many threads as the system can provide
	int NUM_THREADS = omp_get_max_threads();
    omp_set_num_threads(NUM_THREADS);

 
   
	// Set start value for timer
	t = SDL_GetTicks();

	// Generate the Cornell Box
	LoadTestModel( triangles );

	// Every pixel will have a closest intersection
	size_t i;
	float m = std::numeric_limits<float>::max();

	for(i = 0; i < SCREEN_WIDTH*SCREEN_HEIGHT; i++)
	{
		Intersection intersection;
		intersection.distance = m;
		closestIntersection.push_back(intersection);
	}

	R[1][1] = 1.0f;


	while( NoQuitMessageSDL() )
	{
		Update();
		if (updated)
		{
			Draw();
			updated = false;
		}
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );

	return 0;
}

	

