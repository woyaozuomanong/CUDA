#include <stdio.h>

#define LEN 5
struct innerarray
{
float x[LEN];
float y[LEN];
};


int main()
{
	struct innerarray z={{1,2,3,4,5},{1.1,2.2,3.3,4.4,5.5}};
	for(int i=0;i<LEN;i++)
	{
		printf("%f,%f\n",z.x[i],z.y[i]);
	}
	return 0;
}
