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

#define NUM_SCALES   (16)
#define AVG_SECONDS  (3)
#define NEIGH_WEIGHT (0.5)
#define THRES_WEIGHT (4.0)//(6.0)
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
  } scales[NUM_SCALES];

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
        sampleRate = sampleRate<8000? 8000 : sampleRate>384000? 384000 : sampleRate;
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
        fprintf(stderr, "  -r <rate>  -- Use given sampling rate (8000..384000, default 48000).\n");
        fprintf(stderr, "  -n <chars> -- Number of characters to print (1..32, default 8).\n");
        fprintf(stderr, "  -w <hertz> -- Use given bandwith (40..1000, default 170).\n");
        fprintf(stderr, "  -b <baud>  -- Use given baud rate (10..600, default 45.45).\n");
        fprintf(stderr, "  -i         -- Use 16bit signed integer input.\n");
        fprintf(stderr, "  -f         -- Use 32bit floating point input.\n");
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

  // This is how many samples we process at a time
  unsigned int inputStep = sampleRate / (bandWidth / 2);
  // This is how many meaningful bandWidth/2 FFT buckets we have
  unsigned int numChannels = inputStep / 2;
  // This is our baud rate in samples
  unsigned int baudStep = round(sampleRate / baudRate);

  // Allocate FFT plan, input, and output buffers
  fftwf_complex *fftOut = new fftwf_complex[inputStep];
  short *dataIn = new short[inputStep];
  float *fftIn  = new float[inputStep];
  fftwf_plan fft = fftwf_plan_dft_r2c_1d(inputStep, fftIn, fftOut, FFTW_ESTIMATE);

  // Allocate CSDR object storage
  out         = new Csdr::Ringbuffer<unsigned char> *[numChannels];
  outReader   = new Csdr::RingbufferReader<unsigned char> *[numChannels];
  rttyDecoder = new Csdr::BufferedModule<float, unsigned char> *[numChannels];
  bdotDecoder = new Csdr::BufferedModule<unsigned char, unsigned char> *[numChannels];
  snr         = new float[numChannels];

  // RTTY bits are collected here
  int inLevel[numChannels] = {0};
  int inCount[numChannels] = {0};

  // Debug output gets accumulated here
  char dbgOut[numChannels+16];

  // Create CSDR objects
  for(j=0 ; j<numChannels ; ++j)
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
      if(fread(fftIn, sizeof(float), inputStep, inFile) != inputStep) break;
    }
    else
    {
      // Read input data
      if(fread(dataIn, sizeof(short), inputStep, inFile) != inputStep) break;
      // Expand shorts to floats, normalizing them to [-1;1) range
      for(j=0 ; j<inputStep ; ++j)
        fftIn[j] = (float)dataIn[j] / 32768.0;
    }

    // Apply Hamming window
    double hk = 2.0 * M_PI / (inputStep-1);
    for(j=0 ; j<inputStep ; ++j)
      fftIn[j] = fftIn[j] * (0.54 - 0.46 * cos(j * hk));

    // Compute FFT
    fftwf_execute(fft);

    // Go to magnitudes
    for(j=0 ; j<numChannels ; ++j)
      fftOut[j][0] = fftOut[j][1] = sqrt(fftOut[j][0]*fftOut[j][0] + fftOut[j][1]*fftOut[j][1]);

    // Filter out spurs
#if USE_NEIGHBORS
    fftOut[numChannels-1][0] = fmax(0.0, fftOut[numChannels-1][1] - NEIGH_WEIGHT * fftOut[numChannels-2][1]);
    fftOut[0][0] = fmax(0.0, fftOut[0][1] - NEIGH_WEIGHT * fftOut[1][1]);
    for(j=1 ; j<numChannels-1 ; ++j)
      fftOut[j][0] = fmax(0.0, fftOut[j][1] - 0.5 * NEIGH_WEIGHT * (fftOut[j-1][1] + fftOut[j+1][1]));
#endif

    // Sort buckets into scales
    memset(scales, 0, sizeof(scales));
    for(j=0, maxPower=0.0 ; j<numChannels ; ++j)
    {
      float v = fftOut[j][0];
      int scale = floor(log(v));
      scale = scale<0? 0 : scale+1>=NUM_SCALES? NUM_SCALES-1 : scale+1;
      maxPower = fmax(maxPower, v);
      scales[scale].power += v;
      scales[scale].count++;
    }

    // Find most populated scales and use them for ground power
    for(i=0, n=0, accPower=0.0 ; i<NUM_SCALES-1 ; ++i)
    {
      // Look for the next most populated scale
      for(k=i, j=i+1 ; j<NUM_SCALES ; ++j)
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
      if(n>=numChannels/2) break;
    }

//fprintf(stderr, "accPower = %f (%d buckets, %d%%)\n", accPower/n, i+1, 100*n*2/inputStep);

    // Maintain rolling average over AVG_SECONDS
    accPower /= n;
    avgPower += (accPower - avgPower) * inputStep / sampleRate / AVG_SECONDS;

    // Decode by channel
    for(j=0 ; j<numChannels-2 ; ++j)
    {
      float power0 = fftOut[j][0];
      float power1 = fftOut[j+2][0];

      // At least one side has to be above the avgPower * THRES_WEIGHT
      // and the other side has to be RTTY_WEIGHT times less
      int state =
          fmax(power0, power1) < avgPower * THRES_WEIGHT? 0
        : power1 > RTTY_WEIGHT * power0? 1
        : power0 > RTTY_WEIGHT * power1? -1
        : 0;


      // Keep track of the SnR
      power0 = fmax(fmax(power0, power1) / avgPower, 1.0);
      snr[j] += (power0 - snr[j]) * (power0 >= snr[j]? 0.25 : 0.05);

      // Show data by channel, for debugging purposes
      dbgOut[j] = state > 0? '>' : state < 0? '<' : power0 >= THRES_WEIGHT? '=' : '.';

      // Accumulate state data
      n = inCount[j] + inputStep > baudStep? baudStep - inCount[j] : inputStep;
      inCount[j] += n;
      inLevel[j] += state * n;

      // Resync if cannot determine the signal level
      if(abs(inLevel[j]) < inputStep / 2)
//      if(abs(inLevel[j]) + baudStep - inCount[j] < baudStep / 2)
      {
        inCount[j] = inputStep;
        inLevel[j] = state * inputStep;
      }

      // Once enough data accumulated...
      if(inCount[j] < baudStep)
        state = 0;
      else
      {
        // This is our current state
        int lastState = state;
        state = inLevel[j];

        // This is the remaining time
        inCount[j] = inputStep - n;
        inLevel[j] = lastState * inCount[j];

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

        // If collected at least 14 bits...
        if(inReader->available()>=14)
        {
          // ...and these bits are 1xxxxx01xxxxx0...
          float *data = inReader->getReadPointer();
          if((data[0]!=data[6]) && ((data[0]<data[6])!=invert) && (data[7]==data[0]) && (data[13]==data[6]))
          {
            // Process input character
            rttyDecoder[j]->processAll();
            bdotDecoder[j]->processAll();
            // Print output
            printOutput(outFile, j, round((j + 1.5) * bandWidth / 2.0), printChars);
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
  for(j=0 ; j<numChannels-2 ; ++j)
    printOutput(outFile, j, round((j + 1.5) * bandWidth / 2.0), 1);

  // Close files
  if(outFile!=stdout) fclose(outFile);
  if(inFile!=stdin)   fclose(inFile);

  // Release FFTW3 resources
  fftwf_destroy_plan(fft);
  delete [] fftOut;
  delete [] fftIn;
  delete [] dataIn;

  // Release CSDR resources
  for(j=0 ; j<numChannels ; ++j)
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
