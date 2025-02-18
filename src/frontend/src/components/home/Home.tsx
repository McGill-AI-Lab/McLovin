import { Box, Typography, Container } from "@mui/material";
import { motion } from "framer-motion";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";

export const Home = () => {
  const titleVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 1.2,
        ease: [0.6, -0.05, 0.01, 0.99],
      },
    },
  };

  const arrowVariants = {
    initial: { opacity: 0, y: 0 },
    animate: {
      opacity: [0, 1, 1],
      y: [0, 10, 0],
      transition: {
        duration: 2,
        repeat: Infinity,
        repeatType: "reverse",
      },
    },
  };

  return (
    <Container
      maxWidth={false}
      disableGutters
      sx={{
        minHeight: "100vh",
        padding: 0,
        margin: 0,
        width: "100vw",
        background: "linear-gradient(135deg, #ff9a9e, #fad0c4)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Box
        sx={{
          textAlign: "center",
          position: "relative",
        }}
      >
        <motion.div initial="hidden" animate="visible" variants={titleVariants}>
          <Typography
            variant="h1"
            sx={{
              fontSize: { xs: "2.5rem", md: "4rem" },
              color: "white",
              fontWeight: 700,
              textShadow: "2px 2px 4px rgba(0,0,0,0.1)",
              mb: 4,
            }}
          >
            Find Your McGill Match
          </Typography>
        </motion.div>

        <motion.div
          initial="initial"
          animate="animate"
          variants={arrowVariants}
        >
          <KeyboardArrowDownIcon
            sx={{
              fontSize: "3rem",
              color: "white",
              cursor: "pointer",
              "&:hover": {
                transform: "scale(1.1)",
                transition: "transform 0.2s ease-in-out",
              },
            }}
          />
        </motion.div>
      </Box>
    </Container>
  );
};
