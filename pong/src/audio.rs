use std::io::Read;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::BufferSize;
use rubato::{
    Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
pub struct AudioSystem {
    _stream: cpal::Stream,
    sounds: Arc<Mutex<Vec<SoundInstance>>>,
    sample_rate: f64,
}

const BUFFER_SIZE_PER_CHANNEL: usize = 1024;

impl AudioSystem {
    pub fn new() -> eyre::Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");
        println!("Output device: {}", device.name()?);
        let config = device.default_output_config().unwrap();
        println!("Default output config: {:?}", config);
        let mut stream_config = config.config();

        let sample_rate = stream_config.sample_rate.0 as f64;
        stream_config.buffer_size =
            BufferSize::Fixed(BUFFER_SIZE_PER_CHANNEL as u32 * stream_config.channels as u32);

        let sounds = Arc::new(Mutex::new(Vec::<SoundInstance>::new()));
        let sounds_stream = sounds.clone();

        let stream = device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                println!("Audio Thread: requesting {} samples", data.len());
                let mut sounds_lock = sounds_stream.lock().unwrap();
                println!("Audio Thread: Acquired lock");
                for sample in data.iter_mut() {
                    *sample = 0.0;
                }
                for (i, sound) in sounds_lock
                    .iter_mut()
                    .filter(|sound| sound.playing)
                    .enumerate()
                {
                    println!("Sound: {}", i);
                    for (i, sample) in data.iter_mut().enumerate() {
                        let channel = i % stream_config.channels as usize;

                        if sound.resampled_buffer_index >= BUFFER_SIZE_PER_CHANNEL {
                            // Resample
                            let num_samples = sound.resampler.input_frames_next();
                            // Check if we have enough samples
                            if sound.i + num_samples <= sound.sound.samples_channels[0].len() {
                                let (consumed, _) = sound
                                    .resampler
                                    .process_into_buffer(
                                        &[
                                            &sound.sound.samples_channels[0]
                                                [sound.i..sound.i + num_samples],
                                            &sound.sound.samples_channels[1]
                                                [sound.i..sound.i + num_samples],
                                        ],
                                        sound.resampled_buffer.as_mut(),
                                        None,
                                    )
                                    .unwrap();
                                sound.resampled_buffer_index = 0;
                                sound.i += consumed;
                            } else if sound.i <= sound.sound.samples_channels[0].len() {
                                let (consumed, _) = sound
                                    .resampler
                                    .process_partial_into_buffer(
                                        Some(&[
                                            &sound.sound.samples_channels[0][sound.i..],
                                            &sound.sound.samples_channels[1][sound.i..],
                                        ]),
                                        sound.resampled_buffer.as_mut(),
                                        None,
                                    )
                                    .unwrap();
                                sound.resampled_buffer_index = 0;
                                sound.i += consumed;
                            } else {
                                let (consumed, _) = sound
                                    .resampler
                                    .process_partial_into_buffer(
                                        None::<&[&[f32]]>,
                                        sound.resampled_buffer.as_mut(),
                                        None,
                                    )
                                    .unwrap();
                                sound.resampled_buffer_index = 0;
                                sound.i += consumed;
                                if sound.sound.repeat {
                                    sound.i = 0;
                                } else {
                                    sound.playing = false;
                                }
                            }
                        }

                        *sample = sound.resampled_buffer[channel][sound.resampled_buffer_index];

                        if channel == stream_config.channels as usize - 1 {
                            sound.resampled_buffer_index += 1;
                        }
                    }
                }
            },
            move |err| eprintln!("an error occurred on stream: {}", err),
            None,
        )?;
        stream.play()?;

        Ok(Self {
            _stream: stream,
            sounds,
            sample_rate,
        })
    }

    pub fn play_sound(&self, sound: Sound) {
        // Play sound
        let sound_instance = self.create_sound_instance(sound);
        self.sounds.lock().unwrap().push(sound_instance);
    }

    pub fn load_sound<R: Read>(&self, reader: R, repeat: bool) -> eyre::Result<Sound> {
        let reader = hound::WavReader::new(reader)?;

        let spec = reader.spec();
        let samples = match spec.sample_format {
            hound::SampleFormat::Int => match spec.bits_per_sample {
                8 => reader
                    .into_samples::<i8>()
                    .map(|s| s.map(|s| s as f32 / f32::powi(2., 7)))
                    .collect::<Result<Vec<_>, _>>()?,
                16 => reader
                    .into_samples::<i16>()
                    .map(|s| s.map(|s| s as f32 / f32::powi(2., 15)))
                    .collect::<Result<Vec<_>, _>>()?,
                24 => reader
                    .into_samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / f32::powi(2., 23)))
                    .collect::<Result<Vec<_>, _>>()?,
                32 => reader
                    .into_samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / f32::powi(2., 31)))
                    .collect::<Result<Vec<_>, _>>()?,
                _ => return Err(eyre::eyre!("unsupported bits per sample")),
            },
            hound::SampleFormat::Float => {
                if spec.bits_per_sample == 32 {
                    reader
                        .into_samples::<f32>()
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    return Err(eyre::eyre!("unsupported bits per sample"));
                }
            }
        };

        let mut samples_channels: Vec<Vec<f32>> = (0..spec.channels)
            .map(|c| {
                samples
                    .iter()
                    .skip(c as usize)
                    .step_by(spec.channels as usize)
                    .copied()
                    .collect::<Vec<f32>>()
            })
            .collect();
        if samples_channels.len() == 1 {
            // mono sound
            samples_channels.push(samples_channels[0].clone());
        }

        let samples_channels = [samples_channels[0].clone(), samples_channels[1].clone()];

        println!("Samples length: {}", samples.len());

        // Load sound
        Ok(Sound {
            samples_channels: Arc::new(samples_channels),
            sample_rate: spec.sample_rate as f32,
            repeat,
        })
    }

    pub fn create_sound_instance(&self, sound: Sound) -> SoundInstance {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SincFixedOut::<f32>::new(
            self.sample_rate / sound.sample_rate as f64,
            2.0,
            params,
            BUFFER_SIZE_PER_CHANNEL,
            2,
        )
        .unwrap();

        SoundInstance {
            sound,
            resampler,
            resampled_buffer: [
                [0.0; BUFFER_SIZE_PER_CHANNEL],
                [0.0; BUFFER_SIZE_PER_CHANNEL],
            ],
            resampled_buffer_index: BUFFER_SIZE_PER_CHANNEL,
            i: 0,
            playing: true,
        }
    }
}

#[derive(Clone)]
pub struct Sound {
    samples_channels: Arc<[Vec<f32>; 2]>,
    sample_rate: f32,
    repeat: bool,
}

pub struct SoundInstance {
    sound: Sound,
    resampled_buffer: [[f32; BUFFER_SIZE_PER_CHANNEL]; 2],
    resampled_buffer_index: usize,
    resampler: SincFixedOut<f32>,
    i: usize,
    playing: bool,
}
