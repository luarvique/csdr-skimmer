#include "bufmodule.hpp"

#include "csdr/ringbuffer.hpp"
#include "csdr/baudot.hpp"
#include "csdr/rtty.hpp"
#include "fftw3.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define USE_NEIGHBORS  0 // 1: Subtract neighbors from each FFT bucket
#define USE_AVG_BOTTOM 0 // 1: Subtract average value from each bucket
#define USE_AVG_RATIO  1 // 1: Divide each bucket by average value
#define USE_THRESHOLD  0 // 1: Convert each bucket to 0.0/1.0 values
#define USE_TEST       0 // 1: Run test RTTY sequence

#define MAX_SCALES   (16)
#define MAX_INPUT    (sampleRate/(bandWidth/2))
#define MAX_CHANNELS (MAX_INPUT/2)
#define AVG_SECONDS  (3)
#define NEIGH_WEIGHT (0.5)
#define THRES_WEIGHT (6.0)
#define RTTY_WEIGHT  (2.0)

unsigned int sampleRate = 48000; // Input audio sampling rate
unsigned int printChars = 8;     // Number of characters to print at once
unsigned int bandWidth  = 170;   // RTTY bandwidth
float        baudRate   = 45.45; // RTTY baud rate
bool use16bit = false;           // TRUE: Use S16 input values (else F32)
bool showDbg  = false;           // TRUE: Print debug data to stderr
bool invert   = false;           // TRUE: Invert RTTY levels

Csdr::Ringbuffer<unsigned char> **out;
Csdr::RingbufferReader<unsigned char> **outReader;
Csdr::BufferedModule<unsigned char, unsigned char> **bdotDecoder;
Csdr::BufferedModule<float, unsigned char> **rttyDecoder;
float *snr;

#if USE_TEST
static const char *testRtty =
// A      B      C      D      E
  "01100010100111001110101001010100001";
#endif

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
    fprintf(outFile, "%c", p[j]>=' '? p[j] : ' ');

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
  int j, i, k, n, x, y;

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
      case 'w':
        bandWidth = j<argc-1? atoi(argv[++j]) : bandWidth;
        bandWidth = bandWidth<40? 40 : bandWidth>1000? 1000 : bandWidth;
        break;
      case 'b':
        baudRate = j<argc-1? atof(argv[++j]) : baudRate;
        baudRate = baudRate<10.0? 10.0 : baudRate>600.0? 600.0 : baudRate;
        break;
      case 'i':
        use16bit = true;
        break;
      case 'f':
        use16bit = false;
        break;
      case 'x':
        invert = true;
        break;
      case 'd':
        showDbg = true;
        break;
      case 'h':
        fprintf(stderr, "CSDR-Based RTTY Skimmer by Marat Fayzullin\n");
        fprintf(stderr, "Usage: %s [options] [<infile> [<outfile>]]\n", argv[0]);
        fprintf(stderr, "  -r <rate>  -- Use given sampling rate.\n");
        fprintf(stderr, "  -n <chars> -- Number of characters to print.\n");
        fprintf(stderr, "  -i         -- Use 16bit signed integer input.\n");
        fprintf(stderr, "  -f         -- Use 32bit floating point input.\n");
        fprintf(stderr, "  -w <hertz> -- Use given bandwith (170Hz).\n");
        fprintf(stderr, "  -b <baud>  -- Use given baud rate (45.45bd).\n");
        fprintf(stderr, "  -x         -- Exchange meanings of mark and space.\n");
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
  short *dataIn = new short[MAX_INPUT];
  float *fftIn  = new float[MAX_INPUT];
  fftwf_plan fft = fftwf_plan_dft_r2c_1d(MAX_INPUT, fftIn, fftOut, FFTW_ESTIMATE);

  // Allocate CSDR object storage
  out         = new Csdr::Ringbuffer<unsigned char> *[MAX_CHANNELS];
  outReader   = new Csdr::RingbufferReader<unsigned char> *[MAX_CHANNELS];
  rttyDecoder = new Csdr::BufferedModule<float, unsigned char> *[MAX_CHANNELS];
  bdotDecoder = new Csdr::BufferedModule<unsigned char, unsigned char> *[MAX_CHANNELS];
  snr         = new float[MAX_CHANNELS];

  // This is our baud rate in samples
  unsigned int baudStep = floor(2.0 * sampleRate / baudRate);

  // RTTY bits are collected here
  int inLevel[MAX_CHANNELS] = {0};
  int inCount[MAX_CHANNELS] = {0};

  // Debug output gets accumulated here
  char dbgOut[MAX_CHANNELS+16];

  // Create CSDR objects
  for(j=0 ; j<MAX_CHANNELS ; ++j)
  {
    out[j]         = new Csdr::Ringbuffer<unsigned char>(printChars*4);
    outReader[j]   = new Csdr::RingbufferReader<unsigned char>(out[j]);
    rttyDecoder[j] = new Csdr::BufferedModule<float, unsigned char>(new Csdr::RttyDecoder(invert), printChars*4*7);
    bdotDecoder[j] = new Csdr::BufferedModule<unsigned char, unsigned char>(new Csdr::BaudotDecoder(), printChars*4);
    rttyDecoder[j]->connect(bdotDecoder[j]);
    bdotDecoder[j]->setWriter(out[j]);
    snr[j] = 0.0;
  }

  // Read and decode input
  for(avgPower=4.0, x=y=0 ; ; )
  {
    if(!use16bit)
    {
      // Read input data
      if(fread(fftIn, sizeof(float), MAX_INPUT, inFile) != MAX_INPUT) break;
    }
    else
    {
      // Read input data
      if(fread(dataIn, sizeof(short), MAX_INPUT, inFile) != MAX_INPUT) break;
      // Expand shorts to floats, normalizing them to [-1;1) range
      for(j=0 ; j<MAX_INPUT ; ++j)
        fftIn[j] = (float)dataIn[j] / 32768.0;
    }

    // Apply Hamming window
    double hk = 2.0 * M_PI / (MAX_INPUT-1);
    for(j=0 ; j<MAX_INPUT ; ++j)
      fftIn[j] = fftIn[j] * (0.54 - 0.46 * cos(j * hk));

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

#if USE_AVG_RATIO
    // Divide channel signal by the average power
    for(j=0 ; j<MAX_CHANNELS ; ++j)
      fftOut[j][1] = fmax(1.0, fftOut[j][0] / fmax(avgPower, 10.0*FLT_MIN));
#elif USE_AVG_BOTTOM
    // Subtract average power from the channel signal
    for(j=0 ; j<MAX_CHANNELS ; ++j)
      fftOut[j][1] = fmax(0.0, fftOut[j][0] - avgPower);
#elif USE_THRESHOLD
    // Convert channel signal to 1/0 values based on threshold
    for(j=0 ; j<MAX_CHANNELS ; ++j)
      fftOut[j][1] = fftOut[j][0] >= avgPower*THRES_WEIGHT? 1.0 : 0.0;
#endif

#if USE_TEST
    // Run test RTTY sequence on the first decoder
    fftOut[0][1] = (testRtty[x]=='1')!=invert? 0.5 : 5.0;
    fftOut[2][1] = 5.0 - fftOut[0][1];
#endif

    // Decode by channel
    for(j=0 ; j<MAX_CHANNELS-2 ; ++j)
    {
      float power0 = fftOut[j][1];
      float power1 = fftOut[j+2][1];

      int state =
          power1 > RTTY_WEIGHT * power0? 1
        : power0 > RTTY_WEIGHT * power1? -1
        : 0;

      // Keep track of the SnR
      power0 = fmax(fmax(fftOut[j][0], fftOut[j+2][0]) / avgPower, 1.0);
      snr[j] += (power0 - snr[j]) * (power0 >= snr[j]? 0.25 : 0.05);

      // Show data by channel, for debugging purposes
      dbgOut[j] = state > 0? '>' : state < 0? '<' : power1 >= avgPower*THRES_WEIGHT? '=' : '.';

      // Accumulate state data
      n = inCount[j] + MAX_INPUT > baudStep? baudStep - inCount[j] : MAX_INPUT;
      inCount[j] += MAX_INPUT;
      inLevel[j] += state * n;

      // Resync if cannot determine the signal level
      if(abs(inLevel[j]) < MAX_INPUT)
      {
        inCount[j] = MAX_INPUT;
        inLevel[j] = state * MAX_INPUT;
      }

      // Once enough data accumulated...
      if(inCount[j]<baudStep)
        state = 0;
      else
      {
        // This is our current state
        n = state;
        state = inLevel[j];

        // This is the remaining time
        inCount[j] -= baudStep;
        inLevel[j]  = n * inCount[j];

#if USE_TEST
        // Advance test RTTY sequence
        if(!j && !testRtty[++x]) { x = y; y = (y + 1) & 15; }
#endif

        // Show data by channel, for debugging purposes
        dbgOut[j] = state > 0? '1' : state < 0? '0' : '?';
      }

      Csdr::RingbufferReader<float> *inReader = rttyDecoder[j]->rdr();
      Csdr::Ringbuffer<float> *in = rttyDecoder[j]->buf();

      // If there is a valid RTTY bit and channel can accept bits...
      if(state && in->writeable())
      {
        // Fill input buffer with computed bits
        *(in->getWritePointer()) = state>0? 1.0 : 0.0;
        in->advance(1);

        // If collected at least 7 bits...
        if(inReader->available()>=7)
        {
          // ...and these bits are 1xxxxx0...
          float *data = inReader->getReadPointer();
          if((data[0]!=data[6]) && ((data[0]<data[6])!=invert))
          {
            // Process input character
            rttyDecoder[j]->processAll();
            bdotDecoder[j]->processAll();
            // Print output
            printOutput(outFile, j, (int)round((j + 1) * bandWidth), printChars);
          }
          else
          {
            // Skip bit
            inReader->advance(1);
          }
        }
      }
    }

    // Print debug information to the stderr
    dbgOut[j] = '\0';
    if(showDbg)
    {
      fprintf(stderr, "%s (%.2f, %.2f)\n", dbgOut, avgPower, maxPower);
      fflush(stderr);
    }
  }

  // Final printout
  for(j=0 ; j<MAX_CHANNELS-2 ; j++)
    printOutput(outFile, j, (int)round((j + 1) * bandWidth), 1);

  // Close files
  if(outFile!=stdout) fclose(outFile);
  if(inFile!=stdin)   fclose(inFile);

  // Release FFTW3 resources
  fftwf_destroy_plan(fft);
  delete [] fftOut;
  delete [] fftIn;
  delete [] dataIn;

  // Release CSDR resources
  for(j=0 ; j<MAX_CHANNELS ; ++j)
  {
    delete outReader[j];
    delete out[j];
    delete bdotDecoder[j];
    delete rttyDecoder[j];
  }

  // Release CSDR object storage
  delete [] out;
  delete [] outReader;
  delete [] rttyDecoder;
  delete [] bdotDecoder;
  delete [] snr;

  // Done
  return(0);
}
