#include <png.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <eigen3/Eigen/Eigen>


void read_depth_file( std::string & file_name, Eigen::ArrayXXf & depthArray )
{
  char header[8];    // 8 is the maximum size that can be checked

  // open file and test for it being a png
  FILE *fp = fopen(file_name.c_str(), "rb");
  if (!fp)
    std::cout << "[read_png_file] File " << file_name << " could not be opened for reading\n";
  fread(header, 1, 8, fp);
  if( png_sig_cmp( (unsigned char *) header, 0, 8 ) )
    std::cout << "[read_png_file] File " << file_name << " is not recognized as a PNG file\n";

  // initialize stuff
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if( !png_ptr )
    std::cout << "[read_png_file] png_create_read_struct failed\n";

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if( !info_ptr )
    std::cout << "[read_png_file] png_create_info_struct failed\n";

  if( setjmp(png_jmpbuf(png_ptr)) )
    std::cout << "[read_png_file] Error during init_io\n";

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  int width = png_get_image_width(png_ptr, info_ptr);
  int height = png_get_image_height(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  //This is the variable to be returned
  depthArray = Eigen::ArrayXXf(height,width);

  //int number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  //read file
  if( setjmp(png_jmpbuf(png_ptr)) )
    std::cout << "[read_png_file] Error during read_image\n";

  png_bytep * row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  for( int y = 0; y < height; y++ )
    row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

  png_read_image(png_ptr, row_pointers);
  fclose(fp);
  
  int step = width;
  uint16_t * depthImg = (uint16_t *) malloc( height*2*step);

  for( int i = 0; i < height; i++ )
  {
    for( int j = 0; j < width; j++ )
    {
      memcpy( ((unsigned char *) depthImg) + i*2*step + j*2 + 0, row_pointers[i] + j*2 + 1, 1);
      memcpy( ((unsigned char *) depthImg) + i*2*step + j*2 + 1, row_pointers[i] + j*2 + 0, 1);
      float depth = (float) depthImg[i*step+j] / 5000.0f;
      if (depth >= 0.5f && depth <= 3.0f)
        depthArray(i,j) = depth;
      else
        depthArray(i,j) = -1;
    }
  }

  free( (void*) depthImg );
  for( int y = 0; y < height; y++ )
    free( (void*) row_pointers[y] );
}