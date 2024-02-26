fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    pollster::block_on(pong::run())
}
