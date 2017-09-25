#include <iostream>
#include <glm/glm.hpp>
#include "SDL.h"
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <omp.h>

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec2;
using glm::ivec2;
using glm::vec4;
using glm::mat4;

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
/*-------------------------------------------------Structures-----------------------------------------------------------------------*/
struct Vertex
{
public:
	glm::vec3 position;

	Vertex(const Vertex &v0)
	{
		position = v0.position;
	}
	Vertex(){}
};

struct Pixel
{
public:
	int x;
	int y;
	float zinv;
	glm::vec3 pos3d;

	Pixel(int x, int y) 
		: x(x), y(y), zinv(0.0f){}

	Pixel(int x, int y, float zinv) 
		: x(x), y(y), zinv(zinv){}

	Pixel(int x, int y, float zinv, glm::vec3 pos3d) 
		: x(x), y(y), zinv(zinv), pos3d(pos3d){}

	Pixel(){}

};

struct fPixel
{
public:
	float x;
	float y;
	float zinv;
	glm::vec3 pos3d;

	fPixel(float x, float y, float zinv) 
		: x(x), y(y), zinv(zinv){}

	fPixel(float x, float y, float zinv, glm::vec3 pos3d) 
		: x(x), y(y), zinv(zinv), pos3d(pos3d){}

	fPixel(Pixel &p1)
	{
		x = (float) p1.x;
		y = (float) p1.y;
		zinv = p1.zinv;
		pos3d = p1.pos3d;
	}

	fPixel(){}
};


/*------------------------------------------------PIXEL-OPERATORS---------------------------------------------------------*/
Pixel operator-(const Pixel &p1, const Pixel &p2)
{
    return Pixel(p1.x - p2.x, p1.y - p2.y, p1.zinv - p2.zinv, p1.pos3d - p2.pos3d);
}

fPixel operator/(const fPixel &p1, const float div)
{
    return fPixel((float)p1.x / div, (float)p1.y / div, (float) p1.zinv / div, p1.pos3d / div);
}

fPixel operator+=(const fPixel &p1, const fPixel &p2)
{
    return fPixel(p1.x + p2.x, p1.y + p2.y, p1.zinv + p2.zinv);
}


/*---------------------------------------------------------Global-Variables--------------------------------------------------------------------*/

//Screen Declaration
const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
SDL_Surface* screen;
int t;


//Camera Declaration                                                                 
float focalLength = 500.0f;
vec3 cameraPos( 0, 0, -3.001f );
mat3 R = mat3(0.0f);
float yaw = 0;

//Light and colouring declaration 
int numLights = 0;
Light lights[32];
vec3 currentColor;
vec3 currentNormal;
vec3 currentReflectance;
vec3 indirectLight = 0.5f*vec3( 1, 1, 1 );
vec3 pixelColours[SCREEN_WIDTH * SCREEN_HEIGHT];

//Vector triangles declaration
vector<Triangle> triangles;
vector<Triangle> activeTriangles;

//Declare the depth buffer that deals with occlusion
float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

//Buttons                                                      
bool darken_key = false;
bool illuminate_key = false;
bool updated = true;

//Lower & upper bound for Y coordinate
float minY = -1.0f;
float maxY = 1.0f;


/*-------------------------------------------------------FUNCTIONS--------------------------------------------------------------*/
                                                               

void Illuminate(vec3 position, vec3 color, float intensity)
{
	lights[numLights].lightPos = position;
	lights[numLights].lightColor = color;
	lights[numLights].intensity = intensity;

	numLights++;
}

// Compute 2D screen position from 3D position
void VertexShader( const Vertex& v, Pixel& p )
{
	// We need to adjust the vertex positions based on the camera position and rotation
	vec3 pos = (v.position - cameraPos) * R;

	// Store the 3D position of the vertex in the pixel. Divide by the Z value so the value interpolates correctly over the perspective
	p.pos3d = pos / pos.z;

	// Calculate depth of pixel, inversed so the value interpolates correctly over the perspective
	p.zinv = 1.0f / pos.z;

	// Calculate 2D screen position and place (0,0) at top left of the screen
	p.x = int(focalLength * (pos.x * p.zinv)) + (SCREEN_WIDTH / 2.0f);
	p.y = int(focalLength * (pos.y * p.zinv)) + (SCREEN_HEIGHT / 2.0f);
}

// Calculate per pixel lighting
void PixelShader( const Pixel& p , vec3 color, vec3 normal)
{
	int x = p.x;
	int y = p.y;

	// Multiply pixel 3d position by the z value to get the original position from the inverse
	vec3 pPos3d(p.pos3d);
	//pPos3d *= p.pos3d.z;
	pPos3d = pPos3d/p.zinv;
	// Invert the camera transformations of pos3d P'=(P-C)R
	pPos3d = pPos3d*glm::inverse(R);
	pPos3d= pPos3d + cameraPos;

	vec3 result(0.0f, 0.0f, 0.0f);

	for(int i = 0; i < numLights; i++)
	{
		// Apply pos3d transform to light
		vec3 position = lights[i].lightPos;

		// Calculate lighting
		float r = glm::distance(pPos3d, position);
		float A = 4*M_PI*(r*r);
		vec3 color = lights[i].lightColor * lights[i].intensity;
		vec3 rDir = glm::normalize(position - pPos3d);
		vec3 nDir = normal;
		vec3 B = color / A;

		vec3 D = (B * max(glm::dot(rDir,nDir), 0.0f));
		result = result + D;
	}


	vec3 pixelColor = currentReflectance * (result + indirectLight) * color;
	pixelColours[y*SCREEN_HEIGHT + x] = pixelColor;
}

void Bresenham(Pixel a, Pixel b, vector<Pixel>& result)
{
	int x = a.x;
	int y = a.y;
	int dx = b.x-a.x;
	int dy = b.y-a.y;
	int dx2 = 2*dx;
	int dy2 = 2*dy;
	int dydx2 = dy2 - dx2;
	int d = dy2 - dx;

	float zinv = (b.zinv - a.zinv)/float(dx);
	vec3 pos3d = (b.pos3d - a.pos3d)/float(dx);

	for (int i = 0;  i < dx; i++)
	{
		x = x + 1;
		if (d<0)
		{
			d = d + dy2;
		}
		else
		{
			y = y + 1;
			d = d + dydx2;
		}
		if(x >= 0 && x < SCREEN_WIDTH)
		{
			result[i].x = x;
			result[i].y = y;
			result[i].zinv = a.zinv+zinv*float(i);
			result[i].pos3d = a.pos3d+pos3d*float(i);
		}

	}
}

// Draws a line between two points
void DrawLineSDL( SDL_Surface* surface, Pixel a, Pixel b, vec3 color, vec3 normal)
{
	Pixel delta = a - b;
	delta.x = abs(delta.x);
	delta.y = abs(delta.y);
	delta.zinv = abs(delta.zinv);
	delta.pos3d = glm::abs(delta.pos3d);

	// Bresenham
	int pixels = b.x-a.x;
	vector<Pixel> line (pixels);

	Bresenham(a,b,line);
    #pragma omp parallel for
	for(int i = 0; i < pixels; ++i)
	{
		// Ensure pixel is on the screen and is closer to the camera than the current value in the depth buffer
		if(line[i].y < SCREEN_HEIGHT && line[i].y >= 0 && line[i].x < SCREEN_WIDTH && line[i].x >= 0 && line[i].zinv > depthBuffer[line[i].y][line[i].x])
		{
			depthBuffer[line[i].y][line[i].x] = line[i].zinv;
			PixelShader(line[i], color, normal);
		}
	}
}

// Interpolates between two Pixels
void Interpolate( Pixel a, Pixel b, vector<Pixel>& result )
{
	int N = result.size();
	Pixel delta = b-a;

	fPixel step(delta);

	step = (step / float(max(N-1,1)));

	fPixel current( a );

    #pragma omp parallel for 
	for( int i=0; i<N; ++i )
	{
		result[i].x = current.x;
		result[i].y = current.y;
		result[i].zinv = current.zinv;
		result[i].pos3d = current.pos3d;
		current.x = current.x + step.x;
		current.y = current.y + step.y;
		current.zinv = current.zinv + step.zinv;
		current.pos3d = current.pos3d + step.pos3d;
	}
}



void ComputePolygonRows( const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels )
{
	// 1. Find max and min y-value of the polygon
	// and compute the number of rows it occupies.

	int maxY = max(max(vertexPixels[0].y, vertexPixels[1].y), vertexPixels[2].y);
	int minY = min(min(vertexPixels[0].y, vertexPixels[1].y), vertexPixels[2].y);

	int ROWS = maxY - minY + 1;

	// 2. Resize leftPixels and rightPixels
	// so that they have an element for each row.

	leftPixels.resize( ROWS );
	rightPixels.resize( ROWS );

	// 3. Initialize the x-coordinates in leftPixels
	// to some really large value and the x-coordinates
	// in rightPixels to some really small value.

	for( int i = 0; i < ROWS; ++i )
	{
		leftPixels[i].x = +numeric_limits<int>::max();
		rightPixels[i].x = -leftPixels[i].x;
	}

	// 4. Loop through all edges of the polygon and use
	// linear interpolation to find the x-coordinate for
	// each row it occupies. Update the corresponding
	// values in rightPixels and leftPixels.

	for(int i = 0; i < 3; i++)
	{
		int j = (i + 1) % 3; // Ensure all 3 edges are looped through
		// Adjust vertex positions to have y value 0 at minY so Y coordinates map to array indicies
		Pixel v1 (vertexPixels[i].x, vertexPixels[i].y - minY, vertexPixels[i].zinv, vertexPixels[i].pos3d);
		Pixel v2 (vertexPixels[j].x, vertexPixels[j].y - minY, vertexPixels[j].zinv, vertexPixels[j].pos3d);

		int edgePixels = abs(vertexPixels[i].y - vertexPixels[j].y) + 1; // Calculate number of rows this edge occupies
		vector<Pixel> edgeResult(edgePixels); // Create array of ivec2 with number of rows
		Interpolate(v1,v2,edgeResult); // Interpolate between the two vertices

		for(int k = 0; k < edgePixels; k++)
		{
			if(edgeResult[k].x < leftPixels[edgeResult[k].y].x)
			{
				leftPixels[edgeResult[k].y].x = edgeResult[k].x;
				leftPixels[edgeResult[k].y].y = edgeResult[k].y + minY;
				leftPixels[edgeResult[k].y].zinv = edgeResult[k].zinv;
				leftPixels[edgeResult[k].y].pos3d = edgeResult[k].pos3d;
			}

			if(edgeResult[k].x > rightPixels[edgeResult[k].y].x)
			{
				rightPixels[edgeResult[k].y].x = edgeResult[k].x;
				rightPixels[edgeResult[k].y].y = edgeResult[k].y + minY;
				rightPixels[edgeResult[k].y].zinv = edgeResult[k].zinv;
				rightPixels[edgeResult[k].y].pos3d = edgeResult[k].pos3d;
			}
		}
	}
}

// Draw a line for each row of the triangle
void DrawRows( const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels , vec3 color, vec3 normal)
{
	for(unsigned int i = 0; i < leftPixels.size(); i++)
	{
		// If the line is out of frame, don't draw it
		if(!((leftPixels[i].y >= SCREEN_HEIGHT && rightPixels[i].y >= SCREEN_HEIGHT) || (leftPixels[i].y < 0 && rightPixels[i].y < 0)))
		{
			DrawLineSDL(screen, leftPixels[i],rightPixels[i],color, normal);
		}
		else
		{
			continue;
		}

	}
}

void DrawPolygon( const vector<Vertex>& vertices , vec3 color, vec3 normal)
{
	int V = vertices.size();
	vector<Pixel> vertexPixels( V );

	for( int i=0; i<V; ++i )
		VertexShader( vertices[i], vertexPixels[i] );

	vector<Pixel> leftPixels;
	vector<Pixel> rightPixels;

	ComputePolygonRows( vertexPixels, leftPixels, rightPixels );
	DrawRows( leftPixels, rightPixels , color, normal);
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	//Start again by reseting the frame
	for( int y=0; y<SCREEN_HEIGHT; ++y )
	{
		for( int x=0; x<SCREEN_WIDTH; ++x )
		{
			//Set to zero all the values responsible for drawing the frame
			vec3 color( 0.0f, 0.0f, 0.0f );
			depthBuffer[y][x] = 0.0f;
			pixelColours[y*SCREEN_HEIGHT + x] = vec3(0);
			PutPixelSDL( screen, x, y, color );
		}
	}

	// Initialise camera rotation matrix
	vec3 right(R[0][0], R[0][1], R[0][2]);
	vec3 down(R[1][0], R[1][1], R[1][2]);
	vec3 forward(R[2][0], R[2][1], R[2][2]);
	
	//Initialise all keys to be pressed
	Uint8* keystate = SDL_GetKeyState( 0 );

	//Camera moves forward
	if( keystate[SDLK_UP] )
	{
		cameraPos = cameraPos + 0.05f*forward*(dt / 20.0f);
		updated = true;
	}//Camera moves backwards
	else if( keystate[SDLK_DOWN] )
	{
		cameraPos = cameraPos - 0.05f*forward*(dt / 20.0f);
		updated = true;
	}

	//Camera rotates to the left
	if( keystate[SDLK_LEFT] )
	{
		yaw = yaw + 0.01f*(dt / 20.0f);
		updated = 1;
	}//Camera rotates to the right
	else if( keystate[SDLK_RIGHT] )
	{
		yaw = yaw - 0.01f*(dt / 20.0f);
		updated = true;
	}

	// Lights movement
	if (keystate[SDLK_w])
	{
		lights[numLights-1].lightPos.z = lights[numLights-1].lightPos.z + 0.05f*(dt / 20.0f);
		updated = true;
	}
	else if (keystate[SDLK_s])
	{
		lights[numLights-1].lightPos.z = lights[numLights-1].lightPos.z - 0.05f*(dt / 20.0f);
		updated = true;
	}

	if (keystate[SDLK_a])
	{
		lights[numLights-1].lightPos.x = lights[numLights-1].lightPos.x - 0.05f*(dt / 20.0f);
		updated = true;;
	}
	else if (keystate[SDLK_d])
	{
		lights[numLights-1].lightPos.x  = lights[numLights-1].lightPos.x + 0.05f*(dt / 20.0f);
		updated = true;
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

	
	if (updated)
	{
		// Update camera rotation matrix
		float c = cos(yaw);
		float s = sin(yaw);
		R[0][0] = c;
		R[0][2] = s;
		R[2][0] = -s;
		R[2][2] = c;

		vec3 fVec = glm::normalize(vec3(0,0,1.0f)*R);
		float near = cameraPos.z+fVec.z*0.1f;
		float far = cameraPos.z+fVec.z*15.0f;
		float w = (float)SCREEN_WIDTH, h = (float)SCREEN_HEIGHT;

		// Perspective matrix transformation
		mat4 transform = glm::mat4(0.0f);
		// fovy version
		vec3 t(0.0f, -h/2.0f, focalLength);
		vec3 b(0.0f, h/2.0f, focalLength);
		float cy = dot(t,b)/(glm::length(t)*glm::length(b));
		float rfovy = acos(cy);
		float aspect = w/h;
		transform[0][0] = (1.0f/tan(rfovy/2.0f))/aspect;
		transform[1][1] = (1.0f/tan(rfovy/2.0f));
		transform[2][2] = far/(far-near);
		transform[3][2] = near*far/(far-near);
		transform[3][2] = 1.0f;

	
	}
}

void Draw()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	currentReflectance = vec3(1.0f,1.0f,1.0f);
	#pragma omp parallel 
	{

	#pragma omp for nowait
	for( size_t i = 0; i < triangles.size(); ++i )
	{
		if (!triangles[i].isCulled)
		{
			// Get the 3 vertices of the triangle
			vector<Vertex> vertices(3);
			vertices[0].position = triangles[i].v0;
			vertices[1].position = triangles[i].v1;
			vertices[2].position = triangles[i].v2;
			DrawPolygon( vertices , triangles[i].color, triangles[i].normal);
		}
	}
	}
	// Spawn threads
	#pragma omp parallel 
	{

	#pragma omp for nowait
	for (int y = 1; y < SCREEN_HEIGHT - 1; y++)
	{
		for (int x = 1; x < SCREEN_WIDTH - 1; x++)
		{
			vec3 finalColour(0.0f,0.0f,0.0f);
			finalColour = pixelColours[y*SCREEN_HEIGHT+x];

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
	LoadTestModel( triangles );
	

	R[1][1] = 1.01f;

	// Request as many threads as the system can provide
	int NUM_THREADS = omp_get_max_threads();
    omp_set_num_threads(NUM_THREADS);

   
	t = SDL_GetTicks();	// Set start value for timer.

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

