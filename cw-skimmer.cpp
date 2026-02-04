#include "bufmodule.hpp"

#include "csdr/ringbuffer.hpp"
#include "csdr/cw.hpp"
#include "fftw3.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define USE_NEIGHBORS  0 // 1: Subtract neighbors from each FFT bucket
#define USE_AVG_BOTTOM 0 // 1: Subtract average value from each bucket
#define USE_AVG_RATIO  0 // 1: Divide each bucket by average value
#define USE_THRESHOLD  1 // 1: Convert each bucket to 0.0/1.0 values

#define BANDWIDTH    (100)
#define MAX_SCALES   (16)
#define MAX_INPUT    (sampleRate/BANDWIDTH)
#define MAX_CHANNELS (MAX_INPUT/2)
#define AVG_SECONDS  (3)
#define NEIGH_WEIGHT (0.5)
#define THRES_WEIGHT (6.0)

unsigned int sampleRate = 48000; // Input audio sampling rate
unsigned int printChars = 8;     // Number of characters to print at once
bool use16bit = false;           // TRUE: Use S16 input values (else F32)
bool showCw   = false;           // TRUE: Show dits and dahs
bool showDbg  = false;           // TRUE: Print debug data to stderr

Csdr::Ringbuffer<unsigned char> **out;
Csdr::RingbufferReader<unsigned char> **outReader;
Csdr::BufferedModule<float, unsigned char> **cwDecoder;
unsigned int *outState;
float *snr;

// Print output from ith decoder
void printOutput(FILE *outFile, int i, unsigned int freq, unsigned int printChars)
{
  // Must have a minimum of printChars
  int n = outReader[i]->available();
  if(n<printChars) return;

  // Print frequency
  fprintf(outFile, "%d:%d:", freq, (int)(20.0 * log10f(snr[i])));

  // Print characters
  unsigned char *p = outReader[i]->getReadPointer();
  for(int j=0 ; j<n ; ++j)
  {
    switch(outState[i] & 0xFF)
    {
      case '\0':
        // Print character
        fprintf(outFile, "%c", p[j]);
        // Once we encounter a space, wait for stray characters
        if(p[j]==' ') outState[i] = p[j];
        break;
      case ' ':
        // If possible error, save it in state, else print and reset state
        if(strchr("TEI ", p[j])) outState[i] = p[j];
        else
        {
          fprintf(outFile, "%c", p[j]);
          outState[i] = '\0';
        }
        break;
      default:
        // If likely error, skip it, else print and reset state
        if(strchr("TEI ", p[j])) outState[i] = (outState[i]<<8) | p[j];
        else
        {
          for(int k=24 ; k>=0 ; k-=8)
            if((outState[i]>>k) & 0xFF)
              fprintf(outFile, "%c", (outState[i]>>k) & 0xFF);
          fprintf(outFile, "%c", p[j]);
          outState[i] = '\0';
        }
        break;
    }
  }

  // Done printing
  outReader[i]->advance(n);
  printf("\n");
  fflush(outFile);
}

int main(int argc, char *argv[])
{
  FILE *inFile, *outFile;
  const char *inName, *outName;
  float accPower, avgPower, maxPower;
  int j, i, k, n;

  struct
  {
    float power;
    int count;
  } scales[MAX_SCALES];

  // Parse input arguments
  for(j=1, inName=outName=0, inFile=stdin, outFile=stdout ; j<argc ; ++j)
  {
    if(argv[j][0]!='-')
    {
      // First two non-option arguments are filenames
      if(!inName) inName = argv[j];
      else if(!outName) outName = argv[j];
      else
      {
        fprintf(stderr, "%s: Excessive file name '%s'!\n", argv[0], argv[j]);
        return(2);
      }
    }
    else if(strlen(argv[j])!=2)
    {
      // Single-letter options only!
      fprintf(stderr, "%s: Unrecognized option '%s'!\n", argv[0], argv[j]);
      return(2);
    }
    else switch(argv[j][1])
    {
      case 'n':
        printChars = j<argc-1? atoi(argv[++j]) : printChars;
        printChars = printChars<1? 1 : printChars>32? 32 : printChars;
        break;
      case 'r':
        sampleRate = j<argc-1? atoi(argv[++j]) : sampleRate;
        sampleRate = sampleRate<8000? 8000 : sampleRate>48000? 48000 : sampleRate;
        break;
      case 'i':
        use16bit = true;
        break;
      case 'f':
        use16bit = false;
        break;
      case 'd':
        showDbg = true;
        break;
      case 'c':
        showCw = true;
        break;
      case 'h':
        fprintf(stderr, "CSDR-Based CW Skimmer by Marat Fayzullin\n");
        fprintf(stderr, "Usage: %s [options] [<infile> [<outfile>]]\n", argv[0]);
        fprintf(stderr, "  -r <rate>  -- Use given sampling rate.\n");
        fprintf(stderr, "  -n <chars> -- Number of characters to print.\n");
        fprintf(stderr, "  -i         -- Use 16bit signed integer input.\n");
        fprintf(stderr, "  -f         -- Use 32bit floating point input.\n");
        fprintf(stderr, "  -c         -- Print dits and dahs to STDOUT.\n");
        fprintf(stderr, "  -d         -- Print debug information to STDERR.\n");
        fprintf(stderr, "  -h         -- Print this help message.\n");
        return(0);
      default:
        fprintf(stderr, "%s: Unrecognized option '%s'!\n", argv[0], argv[j]);
        return(2);
    }
  }

  // Open input and output files
  inFile = inName? fopen(inName, "rb") : stdin;
  if(!inFile)
  {
    fprintf(stderr, "%s: Failed opening input file '%s'\n", argv[0], inName);
    return(1);
  }
  outFile = outName? fopen(outName, "wb") : stdout;
  if(!outFile)
  {
    fprintf(stderr, "%s: Failed opening output file '%s'\n", argv[0], outName);
    if(inFile!=stdin) fclose(inFile);
    return(1);
  }

  // Allocate FFT plan, input, and output buffers
  fftwf_complex *fftOut = new fftwf_complex[MAX_INPUT];
  short *dataIn  = new short[MAX_INPUT];
  float *dataBuf = new float[MAX_INPUT];
  float *fftIn   = new float[MAX_INPUT];
  fftwf_plan fft = fftwf_plan_dft_r2c_1d(MAX_INPUT, fftIn, fftOut, FFTW_ESTIMATE);

  // Allocate CSDR object storage
  out       = new Csdr::Ringbuffer<unsigned char> *[MAX_CHANNELS];
  outReader = new Csdr::RingbufferReader<unsigned char> *[MAX_CHANNELS];
  cwDecoder = new Csdr::BufferedModule<float, unsigned char> *[MAX_CHANNELS];
  outState  = new unsigned int[MAX_CHANNELS];
  snr       = new float[MAX_CHANNELS];

  // Debug output gets accumulated here
  char dbgOut[MAX_CHANNELS+16];

  // Create and connect CSDR objects, clear output state
  for(j=0 ; j<MAX_CHANNELS ; ++j)
  {
    out[j]       = new Csdr::Ringbuffer<unsigned char>(printChars*4);
    outReader[j] = new Csdr::RingbufferReader<unsigned char>(out[j]);
    cwDecoder[j] = new Csdr::BufferedModule<float, unsigned char>(new Csdr::CwDecoder<float>(sampleRate, showCw), printChars*4);
    cwDecoder[j]->setWriter(out[j]);
    outState[j] = ' ';
    snr[j] = 0.0;
  }

  // Read and decode input
  for(avgPower=4.0 ; ; )
  {
    if(!use16bit)
    {
      // Read input data
      if(fread(dataBuf, sizeof(float), MAX_INPUT, inFile) != MAX_INPUT)
        break;
    }
    else
    {
      // Read input data
      if(fread(dataIn, sizeof(short), MAX_INPUT, inFile) != MAX_INPUT)
        break;
      // Expand shorts to floats, normalizing them to [-1;1) range
      for(j=0 ; j<MAX_INPUT ; ++j)
        dataBuf[j] = (float)dataIn[j] / 32768.0;
    }

    // Apply Hamming window
    double hk = 2.0 * M_PI / (MAX_INPUT-1);
    for(j=0 ; j<MAX_INPUT ; ++j)
      fftIn[j] = dataBuf[j] * (0.54 - 0.46 * cos(j * hk));

    // Compute FFT
    fftwf_execute(fft);

    // Go to magnitudes
    for(j=0 ; j<MAX_CHANNELS ; ++j)
      fftOut[j][0] = fftOut[j][1] = sqrt(fftOut[j][0]*fftOut[j][0] + fftOut[j][1]*fftOut[j][1]);

    // Filter out spurs
#if USE_NEIGHBORS
    fftOut[MAX_CHANNELS-1][0] = fmax(0.0, fftOut[MAX_CHANNELS-1][1] - NEIGH_WEIGHT * fftOut[MAX_CHANNELS-2][1]);
    fftOut[0][0] = fmax(0.0, fftOut[0][1] - NEIGH_WEIGHT * fftOut[1][1]);
    for(j=1 ; j<MAX_CHANNELS-1 ; ++j)
      fftOut[j][0] = fmax(0.0, fftOut[j][1] - 0.5 * NEIGH_WEIGHT * (fftOut[j-1][1] + fftOut[j+1][1]));
#endif

    // Sort buckets into scales
    memset(scales, 0, sizeof(scales));
    for(j=0, maxPower=0.0 ; j<MAX_CHANNELS ; ++j)
    {
      float v = fftOut[j][0];
      int scale = floor(log(v));
      scale = scale<0? 0 : scale+1>=MAX_SCALES? MAX_SCALES-1 : scale+1;
      maxPower = fmax(maxPower, v);
      scales[scale].power += v;
      scales[scale].count++;
    }

    // Find most populated scales and use them for ground power
    for(i=0, n=0, accPower=0.0 ; i<MAX_SCALES-1 ; ++i)
    {
      // Look for the next most populated scale
      for(k=i, j=i+1 ; j<MAX_SCALES ; ++j)
        if(scales[j].count>scales[k].count) k = j;
      // If found, swap with current one
      if(k!=i)
      {
        float v = scales[k].power;
        j = scales[k].count;
        scales[k] = scales[i];
        scales[i].power = v;
        scales[i].count = j;
      }
      // Keep track of the total number of buckets
      accPower += scales[i].power;
      n += scales[i].count;
      // Stop when we collect 1/2 of all buckets
      if(n>=MAX_CHANNELS/2) break;
    }

//fprintf(stderr, "accPower = %f (%d buckets, %d%%)\n", accPower/n, i+1, 100*n*2/MAX_INPUT);

    // Maintain rolling average over AVG_SECONDS
    accPower /= n;
    avgPower += (accPower - avgPower) * MAX_INPUT / sampleRate / AVG_SECONDS;

    // Decode by channel
    for(j=0 ; j<MAX_CHANNELS ; ++j)
    {
      float power = fftOut[j][0];

#if USE_AVG_RATIO
      // Divide channel signal by the average power
      accPower = fmax(1.0, power / fmax(avgPower, 10.0*FLT_MIN));
#elif USE_AVG_BOTTOM
      // Subtract average power from the channel signal
      accPower = fmax(0.0, power - avgPower);
#elif USE_THRESHOLD
      // Convert channel signal to 1/0 values based on threshold
      accPower = power >= avgPower*THRES_WEIGHT? 1.0 : 0.0;
#else
      // Use channel power as-is
      accPower = power;
#endif

      dbgOut[j] = accPower<0.5? '.' : '0' + round(fmax(fmin(accPower / maxPower * 10.0, 9.0), 0.0));

      // Keep track of the SnR
      power = fmax(power / avgPower, 1.0);
      snr[j] += (power - snr[j]) * (power >= snr[j]? 0.2 : 0.05);

      // If CW input buffer can accept samples...
      Csdr::Ringbuffer<float> *in = cwDecoder[j]->buf();
      if(in->writeable()>=MAX_INPUT)
      {
        // Fill input buffer with computed signal power
        float *dst = in->getWritePointer();
        for(i=0 ; i<MAX_INPUT ; ++i) dst[i] = accPower;
        in->advance(MAX_INPUT);

        // Process input for the current channel
        while(cwDecoder[j]->canProcess()) cwDecoder[j]->process();

        // Print output
        printOutput(outFile, j, j * sampleRate / MAX_CHANNELS, printChars);
      }
    }

    // Print debug information to the stderr
    dbgOut[j] = '\0';
    if(showDbg) fprintf(stderr, "%s (%.2f, %.2f)\n", dbgOut, avgPower, maxPower);
  }

  // Final printout
  for(j=0 ; j<MAX_CHANNELS ; j++)
    printOutput(outFile, j, j * sampleRate / MAX_CHANNELS, 1);

  // Close files
  if(outFile!=stdout) fclose(outFile);
  if(inFile!=stdin)   fclose(inFile);

  // Release FFTW3 resources
  fftwf_destroy_plan(fft);
  delete [] fftOut;
  delete [] fftIn;
  delete [] dataBuf;
  delete [] dataIn;

  // Release CSDR resources
  for(j=0 ; j<MAX_CHANNELS ; ++j)
  {
    delete outReader[j];
    delete out[j];
    delete cwDecoder[j];
  }

  // Release CSDR object storage
  delete [] out;
  delete [] outReader;
  delete [] cwDecoder;
  delete [] outState;
  delete [] snr;

  // Done
  return(0);
}
