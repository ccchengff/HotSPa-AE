#pragma once

#define PI (3.14159)
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define DIVUP(x, y) (((x) + (y) -1) / (y))
#define SQUARE(x) ((x) * (x))
#define IS_ODD(x) (((x) &1) == 1)
#define IS_EVEN(x) (((x) &1) == 0)
#define IS_POWER_OF_2(x) ((x) && (((x) & ((x) -1)) == 0))
