fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    pong::run()
}
