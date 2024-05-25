#![windows_subsystem = "windows"]

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    pong::run()
}
